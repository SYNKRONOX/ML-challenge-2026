"""
evaluate.py — macro F0.5 exactly as described on the judging slide.

Per Source-1 entity:  P = |pred ∩ true| / |pred|,  R = |pred ∩ true| / |true|
  both empty            -> 1.0   (correct singleton)
  exactly one empty     -> 0.0
  F0.5 = 1.25 P R / (0.25 P + R)
Score = mean over all Source-1 entities.

Usage:  python code/evaluate.py output/matching_results.tsv data/train_ground_truth.tsv
"""
import sys
from data_io import load_ground_truth


def f05(pred: set, true: set) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def evaluate(pred: dict, truth: dict, verbose=True) -> float:
    scores, sp, sr = [], [], []
    for k, t in truth.items():
        p = pred.get(k, set())
        scores.append(f05(p, t))
        if p:
            sp.append(len(p & t) / len(p))
        if t:
            sr.append(len(p & t) / len(t))
    score = sum(scores) / len(scores)
    if verbose:
        sing = [s for k, s in zip(truth, scores) if not truth[k]]
        print(f"  macro F0.5 = {score:.4f} | mean precision {sum(sp)/max(len(sp),1):.4f} | "
              f"mean recall {sum(sr)/max(len(sr),1):.4f} | singleton acc "
              f"{(sum(sing)/len(sing) if sing else float('nan')):.4f} ({len(sing)} singletons)")
    return score


if __name__ == "__main__":
    pred = load_ground_truth(sys.argv[1])
    truth = load_ground_truth(sys.argv[2])
    evaluate(pred, truth)
