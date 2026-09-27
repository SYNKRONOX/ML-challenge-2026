"""
match.py — Steps 4-5: pair features -> match decisions.

Two interchangeable scorers, both producing a score in [0, 1]:

  RULES (default, no ML)
    name_score = 0.5 * best fuzzy name similarity + 0.5 * IDF-weighted name overlap
                 (fuzzy = max of token-sort, compact ratio, skeleton for transliterated
                  names, partial ratio for website-style names)
    addr_score = 0.5 * IDF-weighted address overlap + 0.5 * address token-sort,
                 + NUM_BONUS when a house number agrees, - NUM_PENALTY when they conflict
    score      = W_NAME * name_score + (1 - W_NAME) * addr_score
                 (transliterated names lean on the address: W_NAME_TRANSLIT)
    special paths:
      * address missing on either side  -> match only if name_score >= NAME_ONLY_THRESHOLD
      * address agreement very specific -> allowed with a weak name (random/garbled names)

  ML (--matcher ml): two stacked gradient-boosted tree models trained on train_ground_truth.
    stage 1: pair features -> p1
    stage 2: pair features + competition features built from p1 -> final score
      s1_max / s1_gap / s1_rank / s1_n50 : best p1 of this Source-1 entity, gap to it, rank of this
                                           pair inside the entity, number of its pairs with p1 > .5
      c_other / c_gap / c_n              : best p1 of any OTHER Source-1 entity for the same S2/S3
                                           record, gap to it, number of S1 entities competing for it
    (learned version of the one-owner rule: a record is claimed by the S1 entity it fits best)
    Same thresholds/one-owner logic afterwards.

Decisions (both scorers):
  1. keep pairs with score >= MATCH_THRESHOLD
  2. one-owner rule: each Source-2/3 record belongs to at most ONE Source-1 entity
     (true in the ground truth — Source 1 is deduplicated) -> keep its best S1 only
  3. relative margin: inside one S1 entity, drop candidates far below its best one
"""
import numpy as np
import pandas as pd

import config
from features import FEATURE_COLS

DEFAULT_PARAMS = {
    "W_NAME": 0.5,
    "W_NAME_TRANSLIT": 0.25,
    "NUM_BONUS": 0.10,
    "NUM_PENALTY": 0.25,
    "NAME_ONLY_THRESHOLD": 0.92,
    "ADDR_ONLY_CONT": 0.95,       # address-only path: containment of address tokens...
    "ADDR_ONLY_IDF": 30.0,        # ...and total rarity of the shared address tokens
    "MATCH_THRESHOLD": 0.70,
    "RELATIVE_MARGIN": 0.20,
}


def rule_scores(f: pd.DataFrame, p: dict) -> np.ndarray:
    tr = f["translit"].values.astype(bool)
    dom = f["domain"].values.astype(bool)
    fuzzy = np.maximum(f["name_sort"].values, f["name_ratio"].values)
    fuzzy = np.where(tr, np.maximum(fuzzy, f["name_skel"].values), fuzzy)
    fuzzy = np.where(dom, np.maximum(fuzzy, f["name_partial"].values), fuzzy)
    name = 0.5 * fuzzy + 0.5 * np.where(tr, np.maximum(f["name_idf_jacc"].values, f["name_skel"].values),
                                        f["name_idf_jacc"].values)

    addr = 0.5 * f["addr_idf_jacc"].values + 0.5 * f["addr_sort"].values
    addr = addr + p["NUM_BONUS"] * (f["num_shared_idf"].values > 0) - p["NUM_PENALTY"] * f["num_conflict"].values
    addr = np.clip(addr, 0, 1)

    w = np.where(tr, p["W_NAME_TRANSLIT"], p["W_NAME"])
    score = w * name + (1 - w) * addr

    addr_only = (f["addr_idf_cont"].values >= p["ADDR_ONLY_CONT"]) & (f["addr_shared_idf"].values >= p["ADDR_ONLY_IDF"])
    score = np.where(addr_only, np.maximum(score, p["MATCH_THRESHOLD"] + 0.01 * addr), score)

    missing = f["addr_missing"].values.astype(bool)
    score = np.where(missing, np.where(name >= p["NAME_ONLY_THRESHOLD"], name, 0.0), score)
    return score.astype(np.float32)


GROUP_COLS = ["p1", "s1_max", "s1_gap", "s1_rank", "s1_n50", "c_other", "c_gap", "c_n"]


def group_features(f: pd.DataFrame, p1: np.ndarray) -> pd.DataFrame:
    """Competition features from stage-1 scores; f needs i1, i2, src."""
    p1 = np.asarray(p1, np.float32)
    i1 = f["i1"].values.astype(np.int64)
    cand = f["i2"].values.astype(np.int64) * 4 + (f["src"].values == "S3")      # unique id per S2/S3 record
    g = pd.DataFrame({"i1": i1, "c": cand, "p": p1})
    out = pd.DataFrame({"p1": p1})
    by1 = g.groupby("i1")["p"]
    out["s1_max"] = by1.transform("max").values
    out["s1_gap"] = p1 - out["s1_max"].values
    out["s1_rank"] = by1.rank(ascending=False, method="first").values.astype(np.float32)
    out["s1_n50"] = g.assign(h=p1 > 0.5).groupby("i1")["h"].transform("sum").values.astype(np.float32)
    # best and second-best p1 per S2/S3 record -> best competitor for each pair
    order = np.lexsort((-p1, cand))
    cs, ps = cand[order], p1[order]
    first = np.r_[True, cs[1:] != cs[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(cs)), 0))
    best = ps[start]
    second_idx = start + 1
    has2 = (second_idx < len(cs)) & (cs[np.minimum(second_idx, len(cs) - 1)] == cs)
    second = np.where(has2, ps[np.minimum(second_idx, len(cs) - 1)], 0.0)
    is_top = np.arange(len(cs)) == start
    other_sorted = np.where(is_top, second, best)
    c_other = np.empty_like(p1)
    c_other[order] = other_sorted
    out["c_other"] = c_other
    out["c_gap"] = p1 - c_other
    out["c_n"] = g.groupby("c")["p"].transform("size").values.astype(np.float32)
    return out.astype(np.float32)


def stage2_matrix(f: pd.DataFrame, p1: np.ndarray) -> np.ndarray:
    return np.hstack([f[FEATURE_COLS].values.astype(np.float32), group_features(f, p1).values])


def stage1_scores(f: pd.DataFrame, model) -> np.ndarray:
    return model["stage1"].predict_proba(f[FEATURE_COLS].values.astype(np.float32))[:, 1].astype(np.float32)


def ml_scores(f: pd.DataFrame, model) -> np.ndarray:
    """Stage 2 only looks at pairs with stage-1 score >= PRUNE_P1 (the rest score 0) —
    identical in training (tune.py), sample runs and the streamed full/test runs."""
    p1 = stage1_scores(f, model)
    if model.get("stage2") is None:
        return p1
    out = np.zeros(len(f), np.float32)
    m = p1 >= config.PRUNE_P1
    if m.any():
        fm = f[m]
        out[m] = model["stage2"].predict_proba(stage2_matrix(fm, p1[m]))[:, 1]
    return out


def decide(f: pd.DataFrame, score: np.ndarray, p: dict) -> pd.DataFrame:
    """f needs columns i1, i2, src. Returns the rows kept as matches (with 'score')."""
    f = f.assign(score=score)
    f = f[f["score"] >= p["MATCH_THRESHOLD"]]
    f = f.sort_values("score", ascending=False).drop_duplicates(["src", "i2"])      # one owner
    best = f.groupby("i1")["score"].transform("max")
    return f[f["score"] >= best - p["RELATIVE_MARGIN"]]
