"""
build_translit.py — learn an Indic-script-token -> English-token dictionary from the TRAIN data.

Source 2/3 often write an Indian business name in its native script
('श्री टेक्नोलॉजी प्राइवेट लिमिटेड' for 'Sree Technology Private Limited'). A generic
transliteration ('shrii tteknolonjii') rarely lines up with the English spelling, but
the vocabulary is small and repetitive, so we learn the mapping from ground-truth pairs:
when a matched record's name has the same number of tokens as its Source-1 name, tokens
are aligned by position and counted. A mapping is kept when it was seen >= MIN_COUNT times
and accounts for >= MIN_SHARE of that token's aligned occurrences.

Only Source-1 entities OUTSIDE the evaluation sample (crc32 % 20 != 0, the entities used by
`--sample 20`) are used, so sample scores stay honest. The same dictionary is used for the
test split (it is learned from training data only; no test labels exist).

  python code/build_translit.py          ->  output/translit_dict.json
"""
import json
import re
import zlib
from collections import Counter, defaultdict

import pandas as pd

import config
from data_io import load_ground_truth, read_tsv

INDIC = re.compile(r"[ऀ-ൿ]")
_TOK = re.compile(r"[^\s,()\[\]/.&+\-]+")
MIN_COUNT, MIN_SHARE = 2, 0.6
EVAL_SAMPLE = 20


def tokens(s: str):
    return _TOK.findall(s)


def main():
    truth = load_ground_truth(config.data_path(config.GROUND_TRUTH_FILE))
    s1 = read_tsv(config.data_path("train_source1.tsv"))[["entity_id", "business_name"]]
    s1 = s1[[zlib.crc32((x + "#sample").encode()) % EVAL_SAMPLE != 0 for x in s1["entity_id"]]]
    owner = {c: s for s in s1["entity_id"] for c in truth.get(s, ())}
    s1_name = dict(zip(s1["entity_id"], s1["business_name"]))
    pair, tok_n = Counter(), Counter()
    for n in (2, 3):
        df = read_tsv(config.data_path(f"train_source{n}.tsv"))[["entity_id", "business_name"]]
        df = df[df["business_name"].str.contains(INDIC)]
        print(f"  source{n}: {len(df):,} names with Indic script")
        for cid, name in zip(df["entity_id"], df["business_name"]):
            sid = owner.get(cid)
            if sid is None:
                continue
            a, b = tokens(name), tokens(s1_name[sid])
            if len(a) != len(b):
                continue
            for x, y in zip(a, b):
                if INDIC.search(x) and not INDIC.search(y):
                    pair[(x, y.lower())] += 1
                    tok_n[x] += 1
    best = defaultdict(lambda: ("", 0))
    for (x, y), c in pair.items():
        if c > best[x][1]:
            best[x] = (y, c)
    d = {x: y for x, (y, c) in best.items() if c >= MIN_COUNT and c / tok_n[x] >= MIN_SHARE}
    path = config.OUTPUT_DIR / "translit_dict.json"
    path.write_text(json.dumps(d, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    print(f"  {len(tok_n):,} Indic tokens aligned, {len(d):,} kept -> {path}")
    for x in list(d)[:15]:
        print(f"    {x} -> {d[x]}")


if __name__ == "__main__":
    main()
