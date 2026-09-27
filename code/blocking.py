"""
blocking.py — Step 2: generate candidate pairs (Source-1 x Source-2/3).

Every record emits several cheap keys; two records become a candidate pair when
they share keys. Several key families are unioned so a record that is noisy in one
field (Hindi-script name, empty address, typo, random replacement name) is still
caught by another:

  n:<token>          name token                         ('n:kga')
  p:<tok>:<tok>      pair of name tokens                ('p:kga:investment' -- names use a small
                                                         vocabulary, so single tokens are often too common)
  f:<compact>        whole compact name                 (also covers 2-letter names like 'om', 'rg')
  k:<skeleton>       phonetic skeleton of a name token  (catches transliterated names)
  c:<prefix6>        first 6 chars of the compact name  ('caangel.com' ~ 'CA Angel')
  a:<num>:<word>     house number + street/area word    ('a:1622:spring')
  aa:<num>:<num>     two numbers of the address         ('aa:216:2000')
  w:<word>:<word>    two address words                  (addresses with no number)

Keys are country-scoped. Keys shared by > MAX_BLOCK_SIZE records of the Source-2/3 side
are dropped (too generic; the S2/S3 side alone decides so a --sample run blocks exactly
like the full run). Each pair gets a name-key score and an address-key score = sum over
shared keys of 1/size**KEY_WEIGHT_POWER (rare keys count far more than common ones).
Per Source-1 entity and source we keep the union of the top TOP_K_NAME by name score,
top TOP_K_ADDR by address score and top TOP_K_TOTAL by the sum -- so a record that
agrees only on the name (empty / different address) is not crowded out by the many
other businesses at the same address, and vice-versa.
"""
import itertools
import numpy as np
import pandas as pd

import config


NAME_FAMILY = ("n", "p", "f", "k", "c")          # first letter of the key type


def record_keys(core, skel, compact, a_text, a_nums, country) -> list:
    c = country + "|"
    keys = set()
    toks = core.split()
    for t in toks:
        if len(t) >= 3:
            keys.add(c + "n:" + t)
    uniq = sorted(set(toks[:6]))
    for t1, t2 in itertools.combinations(uniq, 2):
        keys.add(c + "p:" + t1 + ":" + t2)
    if len(compact) >= 2:
        keys.add(c + "f:" + compact)
    for s in skel.split():
        if len(s) >= 3:
            keys.add(c + "k:" + s)
    if len(compact) >= 4:
        keys.add(c + "c:" + compact[:6])
    words = [w for w in a_text.split() if len(w) >= 3]
    nums = sorted(a_nums.split(), key=len, reverse=True)[:3]
    for x in nums:
        for w in words[:8]:
            keys.add(c + "a:" + x + ":" + w)
    for x1, x2 in itertools.combinations(sorted(nums), 2):
        keys.add(c + "aa:" + x1 + ":" + x2)
    for w1, w2 in itertools.combinations(words[:6], 2):
        keys.add(c + "w:" + w1 + ":" + w2)
    return list(keys)


def _hash(keys) -> np.ndarray:
    """int64 hashes; the lowest bit carries the key family (1 = name key, 0 = address key)."""
    h = pd.util.hash_array(np.array(keys, dtype=object)).view(np.int64)
    fam = np.fromiter((k[k.index("|") + 1] in NAME_FAMILY for k in keys), np.int64, len(keys))
    return (h & ~np.int64(1)) | fam


def _key_table(df: pd.DataFrame) -> pd.DataFrame:
    """Returns DataFrame(key:int64 hash, idx:int32 row position)."""
    hashes, idxs = [], []
    cols = zip(df["n_core"], df["n_skel"], df["n_compact"], df["a_text"], df["a_nums"], df["country"])
    batch_k, batch_i = [], []
    for i, row in enumerate(cols):
        ks = record_keys(*row)
        batch_k.extend(ks)
        batch_i.extend([i] * len(ks))
        if len(batch_k) > 5_000_000:                  # hash in batches to keep memory flat
            hashes.append(_hash(batch_k))
            idxs.append(np.array(batch_i, dtype=np.int32))
            batch_k, batch_i = [], []
    if batch_k:
        hashes.append(_hash(batch_k))
        idxs.append(np.array(batch_i, dtype=np.int32))
    return pd.DataFrame({"key": np.concatenate(hashes), "idx": np.concatenate(idxs)})


class BlockIndex:
    """Key table of one Source-2/3 frame, built once (sorted by key) and probed with Source-1
    chunks via binary search -- no re-hashing of the ~75M-row table for every chunk."""

    def __init__(self, other: pd.DataFrame, label: str = ""):
        self.label = label
        k2 = _key_table(other)
        size2 = k2["key"].value_counts()
        self.good = size2[size2 <= config.MAX_BLOCK_SIZE]
        k2 = k2[k2["key"].isin(self.good.index)].sort_values("key", kind="stable")
        self.keys = k2["key"].values
        self.idx = k2["idx"].values.astype(np.int32)
        w = (1.0 / np.power(self.good.astype(np.float32), config.KEY_WEIGHT_POWER)).astype(np.float32)
        self.w = w.reindex(self.keys).values.astype(np.float32)
        print(f"  [{label}] usable keys: {len(self.good):,}  other key rows: {len(self.keys):,}", flush=True)

    def candidates(self, s1: pd.DataFrame) -> pd.DataFrame:
        """Returns DataFrame(i1, i2, block_score, block_name, block_addr) — positional row
        indices into s1 / other plus the name-key, address-key and total blocking scores."""
        k1 = _key_table(s1)
        lo_all = np.searchsorted(self.keys, k1["key"].values, "left")
        hi_all = np.searchsorted(self.keys, k1["key"].values, "right")
        n_all = hi_all - lo_all
        m = n_all > 0
        k1 = k1[m].assign(lo=lo_all[m], n=n_all[m]).sort_values("idx", kind="stable")
        # chunk Source-1 rows so each expansion produces at most ~PAIR_BUDGET rows (keeps RAM flat)
        load = k1.groupby("idx")["n"].sum()
        bounds, acc, start = [], 0, 0
        for i, v in zip(load.index.values, load.values):
            if acc + v > config.PAIR_BUDGET and acc > 0:
                bounds.append((start, i)); start, acc = i, 0
            acc += v
        bounds.append((start, len(s1)))

        out = []
        idx1_all, lo1_all, n1_all = k1["idx"].values, k1["lo"].values, k1["n"].values
        name1_all = (k1["key"].values & 1).astype(bool)
        for a, b in bounds:
            sel = (idx1_all >= a) & (idx1_all < b)
            n1, lo1 = n1_all[sel], lo1_all[sel]
            if n1.sum() == 0:
                continue
            rep1 = np.repeat(idx1_all[sel], n1)
            # positions lo1[j] .. lo1[j]+n1[j]-1 for every probe j, flattened
            start_of = np.repeat(lo1 - np.r_[0, np.cumsum(n1)[:-1]], n1)
            pos = start_of + np.arange(n1.sum())
            wv = self.w[pos]
            is_name = np.repeat(name1_all[sel], n1)
            pair = (rep1.astype(np.int64) << 32) | self.idx[pos].astype(np.int64)
            g = pd.DataFrame({"pair": pair, "wn": np.where(is_name, wv, 0).astype(np.float32),
                              "wa": np.where(is_name, 0, wv).astype(np.float32)})
            g = g.groupby("pair", sort=False)[["wn", "wa"]].sum().reset_index()
            g["idx1"] = (g["pair"].values >> 32).astype(np.int32)
            g["idx2"] = (g["pair"].values & 0xFFFFFFFF).astype(np.int32)
            g["w"] = g["wn"] + g["wa"]
            keep = np.zeros(len(g), bool)
            for col, k in (("wn", config.TOP_K_NAME), ("wa", config.TOP_K_ADDR), ("w", config.TOP_K_TOTAL)):
                r = g.groupby("idx1", sort=False)[col].rank(method="first", ascending=False).values
                keep |= (r <= k) & (g[col].values > 0)
            out.append(g.loc[keep, ["idx1", "idx2", "w", "wn", "wa"]])
        res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
            {c: np.zeros(0, t) for c, t in (("idx1", np.int32), ("idx2", np.int32), ("w", np.float32),
                                             ("wn", np.float32), ("wa", np.float32))})
        return res.rename(columns={"idx1": "i1", "idx2": "i2", "w": "block_score",
                                   "wn": "block_name", "wa": "block_addr"})


def generate_candidates(s1: pd.DataFrame, other: pd.DataFrame, label: str = "") -> pd.DataFrame:
    return BlockIndex(other, label).candidates(s1)
