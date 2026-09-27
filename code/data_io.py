"""data_io.py — read the tab-separated sources, attach normalized columns, write outputs.

Normalized fields are stored as plain string columns (not Python dicts) so the full
dataset (~12M records) fits in memory:
  n_core, n_skel, n_compact, n_domain, n_translit   (from normalize_name)
  a_text, a_nums                                    (from normalize_address, space-joined)
"""
import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import config
from normalize import normalize_name, normalize_address


def read_tsv(path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, encoding="utf-8")


def add_normalized(df: pd.DataFrame) -> pd.DataFrame:
    core, skel, comp, dom, tr = [], [], [], [], []
    for x in df["business_name"]:
        n = normalize_name(x)
        core.append(n["core"]); skel.append(n["skel"]); comp.append(n["compact"])
        dom.append(n["is_domain"]); tr.append(n["is_translit"])
    at, an = [], []
    for x in df["business_address"]:
        a = normalize_address(x)
        at.append(a["text"]); an.append(a["num_text"])
    df = df[["entity_id", "country"]].copy()
    df["country"] = df["country"].str.lower().str[:2]
    df["n_core"], df["n_skel"], df["n_compact"] = core, skel, comp
    df["n_domain"], df["n_translit"] = np.array(dom, bool), np.array(tr, bool)
    df["a_text"], df["a_nums"] = at, an
    return df


def _code_hash() -> str:
    """Normalized frames depend only on the raw file + these two modules."""
    h = hashlib.md5()
    for path in (Path(__file__).parent / "normalize.py", Path(__file__).parent / "data_io.py",
                 config.OUTPUT_DIR / "translit_dict.json"):
        if path.exists():
            h.update(path.read_bytes())
    return h.hexdigest()[:10]


def load_source(split: str, n: int, data_dir=None) -> pd.DataFrame:
    """Read + normalize one source. Result is cached in output/cache/ (invalidated
    automatically when the raw file, normalize.py or data_io.py change)."""
    if data_dir:
        config.DATA_DIR = data_dir
    path = config.data_path(config.SOURCE_FILE.format(split=split, n=n), split)
    st = path.stat()
    cache = config.OUTPUT_DIR / "cache" / f"{split}_s{n}_{_code_hash()}_{st.st_size}_{int(st.st_mtime)}.pkl"
    if cache.exists():
        df = pd.read_pickle(cache)
        print(f"  source{n}: {len(df):,} rows (cached normalization)", flush=True)
        return df
    df = read_tsv(path)
    print(f"  source{n}: {len(df):,} rows — normalizing...", flush=True)
    df = add_normalized(df)
    cache.parent.mkdir(parents=True, exist_ok=True)
    for old in cache.parent.glob(f"{split}_s{n}_*.pkl"):
        old.unlink()
    df.to_pickle(cache)
    return df


def load_ground_truth(path) -> dict:
    gt = read_tsv(path)
    col_id, col_m = gt.columns[0], gt.columns[1]
    return {r: set(x for x in m.split(",") if x) for r, m in zip(gt[col_id], gt[col_m])}


def write_results(s1_ids, matches: dict, path, header=("source1_entity_id", "matched_entity_ids")):
    """matches: {s1_id: [ids]} ; every S1 id gets a row, empty string = no match."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(i, ",".join(matches.get(i, []))) for i in s1_ids]
    pd.DataFrame(rows, columns=list(header)).to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE)
    print(f"  wrote {path} ({len(rows):,} rows)")
