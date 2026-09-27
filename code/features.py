"""
features.py — Step 3: similarity features for every candidate pair.

String similarities (rapidfuzz, vectorised, 0..1):
  name_tsr      token-set ratio of core names   (word order / extra words)
  name_sort     token-sort ratio of core names  (stricter: extra words DO cost)
  name_ratio    ratio of compact names          (spacing; 'caangel' vs 'ca angel')
  name_partial  partial ratio of compact names  (one name truncated / website form)
  name_skel     token-set ratio of phonetic skeletons (transliterated Indic names)
  addr_tsr / addr_sort   same idea for the address word text

Per-pair loop (run in a process pool):
  name_idf_jacc, name_idf_cont   IDF-weighted shared-token weight / union, / smaller side
  addr_idf_jacc, addr_idf_cont   same for address words + house numbers
  addr_shared_idf                absolute weight of shared address tokens (how specific the agreement is)
  num_shared_idf                 weight of the rarest shared number (0 if none)
  num_conflict                   1 if both sides have numbers and none agree
  num_c1 / num_c2                share of S1's / the other side's numbers found on the other side (-1: none)
  num_reldiff                    closest pair of numbers, |a-b| / max(a,b)   (1: no numbers)
  num_digitsim                   best digit-string similarity of a number pair ('2204' ~ '204', '131' ~ '31')
  tok_un1 / tok_un2              name tokens with no fuzzy counterpart (ratio < .75) on the other side
  tok_un1_idf / tok_un2_idf      rarest such unmatched token -> a real replacement word ('holdings',
                                 'noodles') vs. a typo or legal/decoration word
  tok_align                      mean best fuzzy match of every name token (typo-tolerant overlap)
  unseen2                        share of the other side's name tokens never seen in ANY Source-1 name
                                 (random replacement names 'Novidova', 'Lumfluxdrex'; also garbled typos)
  dom_contain / dom_init         website names: share of the other name's tokens inside the domain;
                                 domain starts with the other name's initials ('siprivate' ~ 'Shyam Investments')
Vectorised lookups:
  name_freq1/2, addr_freq1/2     log count of records (all sources) with exactly this name / address:
                                 a name-only match is only safe when the name is rare
  s1_name_cnt1/2, s1_addr_cnt1   log count of SOURCE-1 entities with this exact name (either side's) /
                                 at this exact address: several S1 owners -> an address-less record is ambiguous
  addr_missing, translit, domain flags
IDF and counts are computed from the provided data only (all Source-1 records of the split + S2 + S3).
"""
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import config


class IDF:
    """Document-frequency tables over name tokens, address tokens, whole names and whole addresses."""

    def __init__(self, s1, others):
        frames = [s1, *others]
        self.name_df, self.addr_df, self.n = Counter(), Counter(), 0
        for df in frames:
            self.n += len(df)
            for s in df["n_core"]:
                self.name_df.update(set(s.split()))
            for w, x in zip(df["a_text"], df["a_nums"]):
                self.addr_df.update(set(w.split()))
                self.addr_df.update("#" + t for t in set(x.split()))
        self.s1_vocab = set()
        for s in s1["n_core"]:
            self.s1_vocab.update(s.split())
        self.log_n = math.log(self.n + 1)
        self.name_count = pd.concat([df["n_compact"] for df in frames]).value_counts()
        self.addr_count = pd.concat([_addr_key(df) for df in frames]).value_counts()
        self.addr_count = self.addr_count[self.addr_count.index != " "]
        self.s1_name_count = s1["n_compact"].value_counts()
        s1_addr = _addr_key(s1).value_counts()
        self.s1_addr_count = s1_addr[s1_addr.index != " "]

    def name(self, t):
        return self.log_n - math.log(1 + self.name_df.get(t, 0))

    def addr(self, t):
        return self.log_n - math.log(1 + self.addr_df.get(t, 0))

    def worker_state(self):
        """The (small) part the pair loop needs, shipped once to every worker."""
        return self.name_df, self.addr_df, self.s1_vocab, self.log_n


def _addr_key(df):
    return df["a_text"].astype(object) + " " + df["a_nums"].astype(object)


def _cp(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=config.WORKERS, dtype=np.float32) / 100.0


# ---------------------------------------------------------------------------------------- pair loop
LOOP_COLS = ["name_idf_jacc", "name_idf_cont", "addr_idf_jacc", "addr_idf_cont", "addr_shared_idf",
             "num_shared_idf", "num_conflict", "addr_missing", "num_c1", "num_c2", "num_reldiff", "num_digitsim",
             "tok_un1", "tok_un2", "tok_un1_idf", "tok_un2_idf", "tok_align", "unseen2", "dom_contain", "dom_init"]
_W = {}


def _init_worker(state):
    _W["name_df"], _W["addr_df"], _W["s1_vocab"], _W["log_n"] = state


def _to_int(x):
    return int(x[:9]) if x.isdigit() else None


def _loop(args):
    core1, core2, at1, at2, an1, an2, comp1, comp2, dom1, dom2 = args
    name_df, addr_df, s1v, log_n = _W["name_df"], _W["addr_df"], _W["s1_vocab"], _W["log_n"]
    ratio = fuzz.ratio
    n = len(core1)
    out = {k: np.zeros(n, np.float32) for k in LOOP_COLS}
    nidf, aidf = {}, {}

    def wn(t):
        v = nidf.get(t)
        if v is None:
            v = nidf[t] = log_n - math.log(1 + name_df.get(t, 0))
        return v

    def wa(t):
        v = aidf.get(t)
        if v is None:
            v = aidf[t] = log_n - math.log(1 + addr_df.get(t, 0))
        return v

    for j in range(n):
        l1, l2 = core1[j].split(), core2[j].split()
        t1, t2 = set(l1), set(l2)
        s1w, s2w = sum(map(wn, t1)), sum(map(wn, t2))
        sh = sum(map(wn, t1 & t2))
        un = s1w + s2w - sh
        out["name_idf_jacc"][j] = sh / un if un else 0
        m = min(s1w, s2w)
        out["name_idf_cont"][j] = sh / m if m else 0

        # fuzzy token alignment
        if t1 and t2:
            b1 = [max(ratio(a, b) for b in t2) for a in t1]
            b2 = [max(ratio(b, a) for a in t1) for b in t2]
            u1 = [a for a, s in zip(t1, b1) if s < 75]
            u2 = [b for b, s in zip(t2, b2) if s < 75]
            out["tok_un1"][j], out["tok_un2"][j] = len(u1), len(u2)
            out["tok_un1_idf"][j] = max(map(wn, u1)) if u1 else 0
            out["tok_un2_idf"][j] = max(map(wn, u2)) if u2 else 0
            out["tok_align"][j] = (sum(b1) + sum(b2)) / (100.0 * (len(b1) + len(b2)))
        if t2:
            out["unseen2"][j] = sum(t not in s1v for t in t2) / len(t2)

        # website-style names
        if dom1[j] or dom2[j]:
            dcomp, otoks = (comp1[j], l2) if dom1[j] else (comp2[j], l1)
            otoks = [t for t in otoks if len(t) >= 2]
            if otoks and dcomp:
                out["dom_contain"][j] = sum(t in dcomp for t in otoks) / len(otoks)
                ini = "".join(t[0] for t in otoks)
                out["dom_init"][j] = float(len(ini) >= 2 and dcomp.startswith(ini))

        # addresses
        w1, w2 = set(at1[j].split()), set(at2[j].split())
        x1, x2 = set(an1[j].split()), set(an2[j].split())
        out["num_c1"][j] = out["num_c2"][j] = -1
        out["num_reldiff"][j] = 1
        if (not w1 and not x1) or (not w2 and not x2):
            out["addr_missing"][j] = 1
            continue
        A1 = w1 | {"#" + t for t in x1}
        A2 = w2 | {"#" + t for t in x2}
        a1w, a2w = sum(map(wa, A1)), sum(map(wa, A2))
        ash = sum(map(wa, A1 & A2))
        aun = a1w + a2w - ash
        out["addr_idf_jacc"][j] = ash / aun if aun else 0
        am = min(a1w, a2w)
        out["addr_idf_cont"][j] = ash / am if am else 0
        out["addr_shared_idf"][j] = ash
        if x1 and x2:
            shared = x1 & x2
            if shared:
                out["num_shared_idf"][j] = max(wa("#" + t) for t in shared)
            else:
                out["num_conflict"][j] = 1
            out["num_c1"][j] = len(shared) / len(x1)
            out["num_c2"][j] = len(shared) / len(x2)
            best_rd, best_ds = 1.0, 0.0
            for a in list(x1)[:6]:
                ia = _to_int(a)
                for b in list(x2)[:6]:
                    ib = _to_int(b)
                    if ia is not None and ib is not None:
                        rd = abs(ia - ib) / max(ia, ib, 1)
                        if rd < best_rd:
                            best_rd = rd
                    ds = ratio(a, b)
                    if ds > best_ds:
                        best_ds = ds
            out["num_reldiff"][j] = best_rd
            out["num_digitsim"][j] = best_ds / 100.0
    return out


_POOL = None


def _pool(idf):
    global _POOL
    if _POOL is None:
        n = config.FEATURE_PROCS or max(1, min(12, (os.cpu_count() or 2) - 2))
        _POOL = (ProcessPoolExecutor(n, initializer=_init_worker, initargs=(idf.worker_state(),)), idf)
    return _POOL[0]


def shutdown_pool():
    global _POOL
    if _POOL is not None:
        _POOL[0].shutdown()
        _POOL = None


def compute_features(pairs: pd.DataFrame, s1: pd.DataFrame, other: pd.DataFrame, idf: IDF) -> pd.DataFrame:
    i1, i2 = pairs["i1"].values, pairs["i2"].values
    g = lambda df, c, idx: np.asarray(df[c].values, dtype=object)[idx]
    core1, core2 = g(s1, "n_core", i1), g(other, "n_core", i2)
    comp1, comp2 = g(s1, "n_compact", i1), g(other, "n_compact", i2)
    sk1, sk2 = g(s1, "n_skel", i1), g(other, "n_skel", i2)
    at1, at2 = g(s1, "a_text", i1), g(other, "a_text", i2)
    an1, an2 = g(s1, "a_nums", i1), g(other, "a_nums", i2)
    dom1, dom2 = s1["n_domain"].values[i1], other["n_domain"].values[i2]

    f = pd.DataFrame({"i1": i1, "i2": i2})
    for c in ("block_score", "block_name", "block_addr"):
        f[c] = pairs[c].values.astype(np.float32)
    f["name_tsr"] = _cp(core1, core2, fuzz.token_set_ratio)
    f["name_sort"] = _cp(core1, core2, fuzz.token_sort_ratio)
    f["name_ratio"] = _cp(comp1, comp2, fuzz.ratio)
    f["name_partial"] = _cp(comp1, comp2, fuzz.partial_ratio)
    f["name_skel"] = _cp(sk1, sk2, fuzz.token_set_ratio)
    f["addr_tsr"] = _cp(at1, at2, fuzz.token_set_ratio)
    f["addr_sort"] = _cp(at1, at2, fuzz.token_sort_ratio)

    n = len(f)
    step = 50_000
    chunks = [(core1[a:a + step], core2[a:a + step], at1[a:a + step], at2[a:a + step], an1[a:a + step],
               an2[a:a + step], comp1[a:a + step], comp2[a:a + step], dom1[a:a + step], dom2[a:a + step])
              for a in range(0, n, step)]
    parts = list(_pool(idf).map(_loop, chunks))
    for k in LOOP_COLS:
        f[k] = np.concatenate([p[k] for p in parts]) if parts else np.zeros(0, np.float32)

    lc = lambda cnt, keys: np.log1p(cnt.reindex(keys).fillna(0).values).astype(np.float32)
    f["name_freq1"], f["name_freq2"] = lc(idf.name_count, comp1), lc(idf.name_count, comp2)
    f["addr_freq1"] = lc(idf.addr_count, at1 + " " + an1)
    f["addr_freq2"] = lc(idf.addr_count, at2 + " " + an2)
    f["s1_name_cnt1"], f["s1_name_cnt2"] = lc(idf.s1_name_count, comp1), lc(idf.s1_name_count, comp2)
    f["s1_addr_cnt1"] = lc(idf.s1_addr_count, at1 + " " + an1)
    f["translit"] = (s1["n_translit"].values[i1] | other["n_translit"].values[i2]).astype(np.int8)
    f["domain"] = (dom1 | dom2).astype(np.int8)
    return f


FEATURE_COLS = ["block_score", "block_name", "block_addr", "name_tsr", "name_sort", "name_ratio", "name_partial",
                "name_skel", "addr_tsr", "addr_sort", *LOOP_COLS,
                "name_freq1", "name_freq2", "addr_freq1", "addr_freq2", "s1_name_cnt1", "s1_name_cnt2",
                "s1_addr_cnt1", "translit", "domain"]
