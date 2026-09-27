"""
run_pipeline.py — end-to-end entity resolution.

  python code/run_pipeline.py --split train --evaluate --sample 20   # build features on 1/20 of Source-1 + score
  python code/tune.py --matcher ml                                   # train model / pick thresholds (cached features)
  python code/run_pipeline.py --split train --evaluate --matcher ml  # full training set (streaming), honest score
  python code/run_pipeline.py --split test --matcher ml              # produce the submission

  --matcher rules|ml   rules = hand-written scorer, ml = stacked gradient boosting (needs tune.py --matcher ml)
  --sample N           (train only) keep 1/N of Source-1 entities; stores all pair features for tune.py
  --blocking-only      (train only) stop after blocking and report blocking recall / pair count
  --stream             score chunk by chunk and keep only promising pairs (automatic without --sample):
                       the full data has ~150M candidate pairs, far too many to hold with all features
  --collect            (train only) build a LARGER training set for tune.py: keep every true pair, every
                       pair the current stage-1 model finds plausible (>= COLLECT_P1) and a random
                       COLLECT_RATE share of the easy negatives with weight 1/COLLECT_RATE

Source 1 is processed in chunks of config.CHUNK_S1 rows against a key index built once per
Source 2/3. Writes to output/:
  candidate_pairs.tsv     blocking set: one row per Source-1 entity, comma-separated candidate ids
  matching_results.tsv    final matches (leaderboard file)
  features_train.pkl      pair features of a --sample run (tune.py re-uses them)
"""
import argparse
import gc
import json
import pickle
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import config
from blocking import BlockIndex
from data_io import load_ground_truth, load_source, write_results
from features import FEATURE_COLS, IDF, compute_features, shutdown_pool
from match import DEFAULT_PARAMS, decide, ml_scores, rule_scores, stage1_scores

SRC_CODE = {"S2": 0, "S3": 1}


def params_path(matcher):
    return config.OUTPUT_DIR / f"best_params_{matcher}.json"


def load_params(matcher) -> dict:
    p = dict(DEFAULT_PARAMS)
    if params_path(matcher).exists():
        p.update(json.loads(params_path(matcher).read_text()))
        print(f"  using tuned params from {params_path(matcher)}")
    else:
        print("  using default params (run tune.py to optimise)")
    return p


def load_model():
    with open(config.OUTPUT_DIR / "model.pkl", "rb") as fh:
        return pickle.load(fh)


def write_candidates(s1_ids, others_ids, i1, i2, src, path):
    """One row per Source-1 entity with its comma-separated candidate ids (streamed to disk)."""
    order = np.lexsort((src, i1))
    i1, i2, src = i1[order], i2[order], src[order]
    bounds = np.searchsorted(i1, np.arange(len(s1_ids) + 1))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        step = 50_000
        for lo in range(0, len(s1_ids), step):
            hi = min(lo + step, len(s1_ids))
            a, b = bounds[lo], bounds[hi]
            ids = np.where(src[a:b] == 0, others_ids["S2"][np.minimum(i2[a:b], len(others_ids["S2"]) - 1)],
                           others_ids["S3"][np.minimum(i2[a:b], len(others_ids["S3"]) - 1)])
            for k in range(lo, hi):
                x, y = bounds[k] - a, bounds[k + 1] - a
                fh.write(f"{s1_ids[k]}\t{','.join(ids[x:y])}\n")
    print(f"  wrote {path} ({len(s1_ids):,} rows, {len(i1):,} candidate pairs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--matcher", choices=["rules", "ml"], default="rules")
    ap.add_argument("--evaluate", action="store_true", help="score against train_ground_truth.tsv")
    ap.add_argument("--sample", type=int, default=0, help="train only: keep 1/N of Source-1 entities")
    ap.add_argument("--blocking-only", action="store_true", help="stop after blocking, report recall")
    ap.add_argument("--stream", action="store_true", help="score per chunk, keep only promising pairs")
    ap.add_argument("--collect", action="store_true", help="train only: weighted training set for tune.py")
    ap.add_argument("--data-dir", default=None, help="folder holding the .tsv files (default: data/)")
    args = ap.parse_args()
    if args.data_dir:
        config.DATA_DIR = Path(args.data_dir)
    if args.split == "test":
        args.sample, args.blocking_only, args.evaluate, args.collect = 0, False, False, False
    if args.collect:
        args.matcher, args.evaluate = "ml", True
    stream = (args.stream or not args.sample or args.collect) and not args.blocking_only
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tag = lambda: f"[{(time.time() - t0) / 60:5.1f} min]"

    print(f"[1/5] loading + normalizing ({args.split}, {'stream' if stream else 'store'} mode)", flush=True)
    s1 = load_source(args.split, 1)
    others = {"S2": load_source(args.split, 2), "S3": load_source(args.split, 3)}
    if not args.blocking_only:
        print("  building IDF / frequency tables", flush=True)
        idf = IDF(s1, list(others.values()))          # always from the FULL Source 1 (sample == full run)
    if args.sample:
        keep = [zlib.crc32((x + "#sample").encode()) % args.sample == 0 for x in s1["entity_id"]]
        s1 = s1[keep].reset_index(drop=True)
        print(f"  sampled Source 1 -> {len(s1):,} rows")
    s1_ids = s1["entity_id"].to_numpy(dtype=object)
    others_ids = {k: v["entity_id"].to_numpy(dtype=object) for k, v in others.items()}

    if stream:
        params = load_params(args.matcher)
        model = load_model() if args.matcher == "ml" else None
        if args.matcher == "ml":
            print(f"  stream mode keeps pairs with stage-1 score >= {config.PRUNE_P1}")
        else:
            print(f"  stream mode keeps pairs with rule score >= MATCH_THRESHOLD ({params['MATCH_THRESHOLD']})")

    if args.collect:
        all_truth = load_ground_truth(config.data_path(config.GROUND_TRUTH_FILE, "train"))
        rng = np.random.default_rng(0)
    cand_i1, cand_i2, cand_src, kept_parts = [], [], [], []
    chunks = [(lo, min(lo + config.CHUNK_S1, len(s1))) for lo in range(0, len(s1), config.CHUNK_S1)]
    for src in list(others):
        other = others.pop(src)
        print(f"{tag()} [2/5] blocking index {src}", flush=True)
        index = BlockIndex(other, src)
        for ci, (lo, hi) in enumerate(chunks):
            pairs = index.candidates(s1.iloc[lo:hi])
            pairs["i1"] = pairs["i1"].values + lo
            cand_i1.append(pairs["i1"].values.astype(np.int32))
            cand_i2.append(pairs["i2"].values.astype(np.int32))
            cand_src.append(np.full(len(pairs), SRC_CODE[src], np.int8))
            msg = f"{tag()} [{src}] S1 chunk {ci + 1}/{len(chunks)} rows {lo:,}-{hi:,}: {len(pairs):,} pairs"
            if args.blocking_only:
                print(msg, flush=True)
                continue
            f = compute_features(pairs, s1, other, idf)
            f["src"] = src
            f["cand_id"] = others_ids[src][f["i2"].values]
            if args.collect:                                   # weighted training sample
                sc = stage1_scores(f, model)
                lab = np.fromiter((c in all_truth.get(x, ()) for x, c in zip(s1_ids[f["i1"].values], f["cand_id"])),
                                  bool, len(f))
                hard = lab | (sc >= config.COLLECT_P1)
                pick = hard | (rng.random(len(f)) < config.COLLECT_RATE)
                f = f.assign(label=lab, weight=np.where(hard, 1.0, 1.0 / config.COLLECT_RATE).astype(np.float32))[pick]
                msg += f", kept {len(f):,} ({lab.sum():,} true)"
            elif stream:                                       # keep only pairs that can still become matches
                sc = stage1_scores(f, model) if model is not None else rule_scores(f, params)
                f = f[sc >= (config.PRUNE_P1 if model is not None else params["MATCH_THRESHOLD"])]
                msg += f", kept {len(f):,}"
            kept_parts.append(f)
            print(msg, flush=True)
            del pairs
        del index, other
        gc.collect()
    shutdown_pool()
    ci1, ci2, csrc = np.concatenate(cand_i1), np.concatenate(cand_i2), np.concatenate(cand_src)
    print(f"{tag()} candidate pairs: {len(ci1):,} ({len(ci1) / max(len(s1), 1):.1f} per S1)", flush=True)

    truth = None
    if args.split == "train" and (args.evaluate or args.blocking_only):
        truth = load_ground_truth(config.data_path(config.GROUND_TRUTH_FILE, "train"))
        truth = {k: truth.get(k, set()) for k in s1_ids}
        true_pairs = sum(len(v) for v in truth.values())
        cid = np.where(csrc == 0, others_ids["S2"][np.minimum(ci2, len(others_ids["S2"]) - 1)],
                       others_ids["S3"][np.minimum(ci2, len(others_ids["S3"]) - 1)])
        found = sum(c in truth[s] for s, c in zip(s1_ids[ci1], cid))
        print(f"  blocking recall (ceiling): {found / max(true_pairs, 1):.4f}  ({found:,}/{true_pairs:,})")
        if args.blocking_only:
            pd.DataFrame({"s1_id": s1_ids[ci1], "cand_id": cid}).to_pickle(config.OUTPUT_DIR / "cand_ids_train.pkl")
            print(f"done in {(time.time() - t0) / 60:.1f} min")
            return
        del cid

    write_candidates(s1_ids, others_ids, ci1, ci2, csrc, config.OUTPUT_DIR / "candidate_pairs.tsv")
    del ci1, ci2, csrc, cand_i1, cand_i2, cand_src
    gc.collect()

    feats = pd.concat(kept_parts, ignore_index=True)
    del kept_parts
    feats["s1_id"] = s1_ids[feats["i1"].values]
    if args.collect:
        feats.to_pickle(config.OUTPUT_DIR / "features_train.pkl")
        pd.Series(s1_ids).to_pickle(config.OUTPUT_DIR / "s1_ids_train.pkl")
        print(f"  saved weighted training set: {len(feats):,} pairs, {feats['label'].sum():,} true, "
              f"effective {feats['weight'].sum():,.0f} -> run: python code/tune.py --matcher ml --eval-sample 20")
        print(f"done in {(time.time() - t0) / 60:.1f} min")
        return
    if not stream:
        feats.to_pickle(config.OUTPUT_DIR / f"features_{args.split}.pkl")
        pd.Series(s1_ids).to_pickle(config.OUTPUT_DIR / f"s1_ids_{args.split}.pkl")

    print(f"{tag()} [4/5] matching ({args.matcher}) on {len(feats):,} pairs", flush=True)
    params = load_params(args.matcher)
    if args.matcher == "ml":
        if not (config.OUTPUT_DIR / "model.pkl").exists():
            raise SystemExit("no output/model.pkl — run: python code/tune.py --matcher ml")
        kept = decide(feats, ml_scores(feats, load_model()), params)
    else:
        kept = decide(feats, rule_scores(feats, params), params)
    matches = kept.groupby("s1_id")["cand_id"].apply(list).to_dict()
    write_results(s1_ids, matches, config.OUTPUT_DIR / "matching_results.tsv")

    if truth is not None:
        print(f"{tag()} [5/5] evaluating")
        from evaluate import evaluate
        evaluate({k: set(v) for k, v in matches.items()}, truth)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
