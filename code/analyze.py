"""
analyze.py — error analysis on the training sample (run after run_pipeline.py --split train --sample N).

  python code/analyze.py --sample 20 blocking      # true pairs missed by blocking, grouped by cause
  python code/analyze.py --sample 20 matching      # false positives / false negatives of the matcher

Uses output/features_train.pkl + cached normalized sources; prints raw records for inspection.
"""
import argparse
import json
import random
import zlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import config
from blocking import record_keys
from data_io import load_ground_truth, load_source, read_tsv
from match import DEFAULT_PARAMS, decide, rule_scores

KEY_COLS = ["n_core", "n_skel", "n_compact", "a_text", "a_nums", "country"]


def sampled_s1(sample):
    s1 = load_source("train", 1)
    if sample:
        keep = [zlib.crc32((x + "#sample").encode()) % sample == 0 for x in s1["entity_id"]]
        s1 = s1[keep].reset_index(drop=True)
    return s1


def raw_lookup(ids):
    """{id: 'name | address'} for the given ids, read from the raw files."""
    ids, out = set(ids), {}
    for n in (1, 2, 3):
        df = read_tsv(config.data_path(config.SOURCE_FILE.format(split="train", n=n)))
        df = df[df["entity_id"].isin(ids)]
        out.update({i: f"{a} | {b}" for i, a, b in zip(df["entity_id"], df["business_name"], df["business_address"])})
    return out


def blocking_misses(args):
    # candidate set of the most recent run (blocking-only run or full run, whichever is newer)
    paths = [config.OUTPUT_DIR / n for n in ("cand_ids_train.pkl", "features_train.pkl") if (config.OUTPUT_DIR / n).exists()]
    path = max(paths, key=lambda x: x.stat().st_mtime)
    print(f"candidates from {path.name}")
    feats = pd.read_pickle(path)[["s1_id", "cand_id"]]
    s1 = sampled_s1(args.sample)
    truth = load_ground_truth(config.data_path(config.GROUND_TRUTH_FILE))
    cand = set(zip(feats["s1_id"], feats["cand_id"]))
    missed = [(a, b) for a in s1["entity_id"] for b in truth.get(a, ()) if (a, b) not in cand]
    total = sum(len(truth.get(a, ())) for a in s1["entity_id"])
    print(f"true pairs {total:,}  missed by blocking {len(missed):,}  recall {1 - len(missed) / total:.4f}")

    rec = {}
    s1i = s1.set_index("entity_id")
    for a, _ in missed:
        rec[a] = s1i.loc[a, KEY_COLS].tolist()
    by_src = defaultdict(list)
    for a, b in missed:
        by_src[b[:2]].append((a, b))
    key_sizes = {}
    for n in (2, 3):
        other = load_source("train", n)
        want = {b for _, b in by_src[f"S{n}"]}
        oi = other[other["entity_id"].isin(want)].set_index("entity_id")
        for b in want:
            rec[b] = oi.loc[b, KEY_COLS].tolist()
        # sizes of the keys shared by missed pairs (count in the other source)
        shared_all = set()
        for a, b in by_src[f"S{n}"]:
            shared_all |= set(record_keys(*rec[a])) & set(record_keys(*rec[b]))
        cnt = Counter()
        for row in zip(*(other[c] for c in KEY_COLS)):
            for k in record_keys(*row):
                if k in shared_all:
                    cnt[k] += 1
        key_sizes[n] = cnt
        del other

    causes, examples = Counter(), defaultdict(list)
    for a, b in missed:
        ra, rb = rec[a], rec[b]
        shared = set(record_keys(*ra)) & set(record_keys(*rb))
        n = int(b[1])
        if not shared:
            if ra[5] != rb[5]:
                c = "no key: country differs"
            elif not (ra[3] or ra[4]) or not (rb[3] or rb[4]):
                c = "no key: address missing" + (" + name differs" if not set(ra[0].split()) & set(rb[0].split()) else "")
            elif not set(ra[0].split()) & set(rb[0].split()):
                c = "no key: no shared name token"
            else:
                c = "no key: other"
        elif all(key_sizes[n][k] > config.MAX_BLOCK_SIZE for k in shared):
            c = "keys only in oversized blocks"
        else:
            c = "cut by top-K cap"
        causes[c] += 1
        examples[c].append((a, b))
    print("\nmissed pairs by cause:")
    for c, k in causes.most_common():
        print(f"  {k:6,}  {100 * k / len(missed):5.1f}%  {c}")
    random.seed(0)
    show = {c: random.sample(v, min(args.n, len(v))) for c, v in examples.items()}
    raw = raw_lookup({x for v in show.values() for p in v for x in p})
    for c, v in show.items():
        print(f"\n=== {c}")
        for a, b in v:
            print(f"  {raw.get(a)}\n    -> {raw.get(b)}")
            print(f"       norm: {rec[a][0]!r} / {rec[a][3]!r} {rec[a][4]!r}  ||  {rec[b][0]!r} / {rec[b][3]!r} {rec[b][4]!r}")


def matching_errors(args):
    f = pd.read_pickle(config.OUTPUT_DIR / "features_train.pkl")
    truth = load_ground_truth(config.data_path(config.GROUND_TRUTH_FILE))
    f["label"] = np.fromiter((c in truth.get(s, ()) for s, c in zip(f["s1_id"], f["cand_id"])), bool, len(f))
    p = dict(DEFAULT_PARAMS)
    pp = config.OUTPUT_DIR / f"best_params_{args.matcher}.json"
    if pp.exists():
        p.update(json.loads(pp.read_text()))
    if args.matcher == "ml":                     # out-of-fold scores written by tune.py --matcher ml
        f["score"] = np.load(config.OUTPUT_DIR / "oof_ml.npy")
    else:
        f["score"] = rule_scores(f, p)
    kept = decide(f, f["score"].values, p)
    kset = set(zip(kept["s1_id"], kept["cand_id"]))
    f["kept"] = [x in kset for x in zip(f["s1_id"], f["cand_id"])]
    fp = f[f["kept"] & ~f["label"]]
    fn = f[~f["kept"] & f["label"]]
    print(f"kept {len(kept):,}  FP {len(fp):,}  FN (in candidates) {len(fn):,}")
    sing = {s for s in pd.read_pickle(config.OUTPUT_DIR / "s1_ids_train.pkl") if not truth.get(s)}
    print(f"FP pairs on true singletons: {fp['s1_id'].isin(sing).sum():,}")
    def cause(d):
        c = np.full(len(d), "other", object)
        rules = [  # first match wins, most specific first
            ("address missing", d["addr_missing"] == 1),
            ("transliterated name", d["translit"] == 1),
            ("website-style name", d["domain"] == 1),
            ("random/unrelated name (tsr<.5)", d["name_tsr"] < 0.5),
            ("house-number conflict", d["num_conflict"] == 1),
            ("look-alike name, same address", (d["addr_idf_jacc"] >= 0.5) & (d["name_tsr"] < 0.9)),
            ("same name, different address", (d["name_tsr"] >= 0.9) & (d["addr_idf_jacc"] < 0.5)),
            ("near-identical (name & address)", (d["name_tsr"] >= 0.9) & (d["addr_idf_jacc"] >= 0.5)),
        ]
        for lab, m in reversed(rules):
            c[m.values] = lab
        return pd.Series(c, index=d.index)
    for name, df in (("FP", fp), ("FN", fn)):
        vc = cause(df).value_counts()
        print(f"\n{name} by cause:")
        for k, v in vc.items():
            print(f"  {v:7,}  {100 * v / len(df):5.1f}%  {k}")
    cols = ["score", "name_tsr", "name_sort", "name_skel", "name_idf_jacc", "addr_idf_jacc", "addr_sort",
            "num_shared_idf", "num_conflict", "addr_missing", "translit", "domain"]
    fp, fn = fp.assign(cause=cause(fp)), fn.assign(cause=cause(fn))
    show = pd.concat([d.groupby("cause", group_keys=False)[d.columns.tolist()]
                      .apply(lambda x: x.sample(min(args.n, len(x)), random_state=0)).assign(kind=k)
                      for k, d in (("FALSE POSITIVE", fp), ("FALSE NEGATIVE", fn))])
    raw = raw_lookup(set(show["s1_id"]) | set(show["cand_id"]))
    for (kind, c), grp in show.groupby(["kind", "cause"], sort=False):
        print(f"\n===== {kind}: {c}")
        for r in grp.itertuples():
            print(f"  {raw.get(r.s1_id)}\n    -> {raw.get(r.cand_id)}")
            print("       " + " ".join(f"{c}={getattr(r, c):.2f}" for c in cols))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["blocking", "matching"])
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("-n", type=int, default=12, help="examples to print per group")
    ap.add_argument("--matcher", choices=["rules", "ml"], default="ml")
    a = ap.parse_args()
    blocking_misses(a) if a.what == "blocking" else matching_errors(a)
