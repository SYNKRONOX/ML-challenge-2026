"""
tune.py — choose thresholds (and, for --matcher ml, train the model) on training data.

Run AFTER `run_pipeline.py --split train`: it re-uses output/features_train.pkl,
so each trial takes seconds. Saves output/best_params_<matcher>.json (and
output/model.pkl for ml), which run_pipeline.py picks up automatically.

  python code/tune.py                  # rules
  python code/tune.py --matcher ml     # gradient boosting (2-stage stacked)
  python code/tune.py --matcher ml --eval-sample 20   # after run_pipeline --collect: train on the big
                                                      # weighted set, report on the held-out sample-20 entities
"""
import argparse
import json
import pickle
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data_io import load_ground_truth
from features import FEATURE_COLS
from match import DEFAULT_PARAMS, decide, rule_scores, stage2_matrix

RULE_GRID = {
    "MATCH_THRESHOLD": [0.55, 0.6, 0.63, 0.66, 0.68, 0.7, 0.72, 0.74, 0.77, 0.8],
    "W_NAME": [0.3, 0.4, 0.5, 0.6, 0.7],
    "W_NAME_TRANSLIT": [0.1, 0.2, 0.25, 0.3, 0.4],
    "NUM_BONUS": [0.0, 0.05, 0.1, 0.2],
    "NUM_PENALTY": [0.0, 0.1, 0.25, 0.4],
    "NAME_ONLY_THRESHOLD": [0.8, 0.85, 0.88, 0.9, 0.92, 0.95, 0.98, 1.01],
    "ADDR_ONLY_CONT": [0.8, 0.9, 0.95, 1.0, 1.01],
    "ADDR_ONLY_IDF": [15, 20, 25, 30, 40, 60],
    "RELATIVE_MARGIN": [0.05, 0.1, 0.15, 0.2, 0.3, 1.0],
}
DECISION_GRID = {
    "MATCH_THRESHOLD": [round(x, 2) for x in np.arange(0.2, 0.95, 0.05)],
    "RELATIVE_MARGIN": [0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
}


def macro_f05(kept, s1_ids, ntrue):
    g = kept.groupby("s1_id").agg(tp=("label", "sum"), npred=("label", "size")).reindex(s1_ids, fill_value=0)
    tp, npred = g["tp"].values, g["npred"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        P, R = tp / np.maximum(npred, 1), tp / np.maximum(ntrue, 1)
        F = np.where(tp > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
    return np.where((npred == 0) & (ntrue == 0), 1.0, F).mean()


def breakdown(kept, s1_ids, ntrue):
    """Same quantities as evaluate.py: mean precision / recall over entities with
    predictions / truths, singleton accuracy, macro F0.5."""
    g = kept.groupby("s1_id").agg(tp=("label", "sum"), npred=("label", "size")).reindex(s1_ids, fill_value=0)
    tp, npred = g["tp"].values, g["npred"].values
    P = tp[npred > 0] / npred[npred > 0]
    R = tp[ntrue > 0] / ntrue[ntrue > 0]
    sing = ntrue == 0
    return (f"macro F0.5 = {macro_f05(kept, s1_ids, ntrue):.4f} | mean precision {P.mean():.4f} | "
            f"mean recall {R.mean():.4f} | singleton acc {(npred[sing] == 0).mean():.4f} ({sing.sum()} singletons)")


def coord_ascent(grid, params, score_fn, rounds=2):
    best, best_s = dict(params), score_fn(params)
    print(f"  start F0.5={best_s:.4f}")
    for r in range(rounds):
        for k, vals in grid.items():
            for v in vals:
                s = score_fn(dict(best, **{k: v}))
                if s > best_s + 1e-5:
                    best, best_s = dict(best, **{k: v}), s
            print(f"  round {r + 1} {k:20s} = {best[k]!s:6}  F0.5={best_s:.4f}", flush=True)
    return best, best_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matcher", choices=["rules", "ml"], default="rules")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--stages", type=int, choices=[1, 2], default=2, help="ml: 1 = single model, 2 = stacked")
    ap.add_argument("--eval-sample", type=int, default=0,
                    help="score/tune only on Source-1 entities of this --sample (e.g. 20 after --collect)")
    args = ap.parse_args()
    if args.data_dir:
        config.DATA_DIR = Path(args.data_dir)

    f = pd.read_pickle(config.OUTPUT_DIR / "features_train.pkl")
    s1_ids = pd.Index(pd.read_pickle(config.OUTPUT_DIR / "s1_ids_train.pkl"))
    truth = load_ground_truth(config.data_path(config.GROUND_TRUTH_FILE, "train"))
    if "label" not in f:
        f["label"] = np.fromiter((c in truth.get(s, ()) for s, c in zip(f["s1_id"], f["cand_id"])), bool, len(f))
    w = f["weight"].values if "weight" in f else np.ones(len(f), np.float32)
    # entities the scores are reported / thresholds tuned on (--eval-sample 20: the entities held
    # out of build_translit.py, so the transliteration dictionary never saw their pairs)
    if args.eval_sample:
        s1_ids = s1_ids[[zlib.crc32((x + "#sample").encode()) % args.eval_sample == 0 for x in s1_ids]]
    ev = f["s1_id"].isin(s1_ids).values
    ntrue = np.array([len(truth.get(k, ())) for k in s1_ids])
    print(f"{len(f):,} candidate pairs ({w.sum():,.0f} weighted), {f['label'].sum():,} true, "
          f"{len(s1_ids):,} Source-1 entities evaluated")

    if args.matcher == "rules":
        fe = f[ev]
        best, s = coord_ascent(RULE_GRID, DEFAULT_PARAMS,
                               lambda p: macro_f05(decide(fe, rule_scores(fe, p), p), s1_ids, ntrue))
        print("  " + breakdown(decide(fe, rule_scores(fe, best), best), s1_ids, ntrue))
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        mk = lambda: HistGradientBoostingClassifier(**config.HGB_PARAMS)
        # 2-fold by Source-1 entity -> honest out-of-fold scores for threshold tuning
        fold = np.fromiter(((zlib.crc32(x.encode()) >> 11) & 1 for x in f["s1_id"]), np.int8, len(f))
        y = f["label"].values

        def oof_of(X, yy, ww, ff, tag):
            oof = np.zeros(len(yy), np.float32)
            for k in (0, 1):
                m = mk().fit(X[ff != k], yy[ff != k], sample_weight=ww[ff != k])
                oof[ff == k] = m.predict_proba(X[ff == k])[:, 1]
                print(f"  {tag} fold {k} trained ({m.n_iter_} iters)", flush=True)
            return oof

        fe = f[ev]
        X1 = f[FEATURE_COLS].values.astype(np.float32)
        oof1 = oof_of(X1, y, w, fold, "stage1")
        print("  stage-1 out-of-fold " + breakdown(decide(fe, oof1[ev], DEFAULT_PARAMS), s1_ids, ntrue))
        # stage 2: competition features from OUT-OF-FOLD stage-1 scores, only on pairs that survive
        # the stage-1 pruning floor (exactly what the streamed full/test runs see)
        live = oof1 >= config.PRUNE_P1
        print(f"  stage-2 pairs (stage-1 >= {config.PRUNE_P1}): {live.sum():,} of {len(f):,}, "
              f"true pairs lost by pruning: {(y & ~live).sum():,}", flush=True)
        X2 = stage2_matrix(f[live], oof1[live])
        oof = oof1.copy()
        if args.stages == 2:
            oof[:] = 0
            oof[live] = oof_of(X2, y[live], w[live], fold[live], "stage2")
        oof_e = oof[ev]
        best, s = coord_ascent(DECISION_GRID, DEFAULT_PARAMS,
                               lambda p: macro_f05(decide(fe, oof_e, p), s1_ids, ntrue))
        print("  out-of-fold " + breakdown(decide(fe, oof_e, best), s1_ids, ntrue))
        np.save(config.OUTPUT_DIR / "oof_ml.npy", oof)             # analyze.py uses these
        model = {"stage1": mk().fit(X1, y, sample_weight=w),
                 "stage2": mk().fit(X2, y[live], sample_weight=w[live]) if args.stages == 2 else None}
        with open(config.OUTPUT_DIR / "model.pkl", "wb") as fh:
            pickle.dump(model, fh)
        print(f"  saved {config.OUTPUT_DIR / 'model.pkl'}")
    path = config.OUTPUT_DIR / f"best_params_{args.matcher}.json"
    path.write_text(json.dumps(best, indent=2))
    print(f"best macro F0.5 on train = {s:.4f}\nsaved {path}")


if __name__ == "__main__":
    main()
