"""
check_submission.py — format checks for the deliverables (the challenge's utils/validate_submission.py
was not provided, so this re-implements the obvious rules).

  python code/check_submission.py                 # checks output/*.tsv against the test split
  python code/check_submission.py --split train   # (after a train run)

matching_results.tsv : header 'source1_entity_id<TAB>matched_entity_ids'; exactly one row per Source-1
                       entity of the split; matched ids comma-separated (empty for singletons), each an
                       existing S2-/S3- id of the split, no id repeated in a row, no id used by two rows
candidate_pairs.tsv  : header 'source1_entity_id<TAB>candidate_entity_ids'; one row per Source-1 entity;
                       every matched id must also be a candidate of the same entity
"""
import argparse
import re
import sys

import config
from data_io import read_tsv

ID_RE = {"S1": re.compile(r"^S1-\d+$"), "S2": re.compile(r"^S2-\d+$"), "S3": re.compile(r"^S3-\d+$")}


def load(path, header):
    with open(path, encoding="utf-8") as fh:
        first = fh.readline().rstrip("\n")
        if first != "\t".join(header):
            return None, [f"{path.name}: header is {first!r}, expected {chr(9).join(header)!r}"]
        rows, errs = {}, []
        for n, line in enumerate(fh, 2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                errs.append(f"{path.name}:{n}: expected 2 tab-separated columns, got {len(parts)}")
                continue
            if parts[0] in rows:
                errs.append(f"{path.name}:{n}: duplicate Source-1 id {parts[0]}")
            rows[parts[0]] = [x for x in parts[1].split(",") if x] if parts[1] else []
            if parts[1] and any(not x for x in parts[1].split(",")):
                errs.append(f"{path.name}:{n}: empty id inside the comma list")
        return rows, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test"])
    args = ap.parse_args()
    src = {n: set(read_tsv(config.data_path(config.SOURCE_FILE.format(split=args.split, n=n), args.split))["entity_id"])
           for n in (1, 2, 3)}
    others = src[2] | src[3]
    errs = []

    res, e = load(config.OUTPUT_DIR / "matching_results.tsv", ("source1_entity_id", "matched_entity_ids"))
    errs += e
    cand, e = load(config.OUTPUT_DIR / "candidate_pairs.tsv", ("source1_entity_id", "candidate_entity_ids"))
    errs += e
    if res is not None:
        missing, extra = src[1] - res.keys(), res.keys() - src[1]
        if missing:
            errs.append(f"matching_results: {len(missing):,} Source-1 ids missing (e.g. {sorted(missing)[:3]})")
        if extra:
            errs.append(f"matching_results: {len(extra):,} unknown Source-1 ids (e.g. {sorted(extra)[:3]})")
        bad_fmt = [k for k in res if not ID_RE["S1"].match(k)]
        if bad_fmt:
            errs.append(f"matching_results: {len(bad_fmt):,} malformed Source-1 ids (e.g. {bad_fmt[:3]})")
        owner, n_match = {}, 0
        for k, v in res.items():
            if len(set(v)) != len(v):
                errs.append(f"matching_results: {k} lists an id twice")
            for x in v:
                n_match += 1
                if x not in others:
                    errs.append(f"matching_results: {k} -> {x} is not an S2/S3 id of the {args.split} split")
                if x in owner and owner[x] != k:
                    errs.append(f"matching_results: {x} matched to both {owner[x]} and {k}")
                owner[x] = k
        sing = sum(1 for v in res.values() if not v)
        print(f"matching_results.tsv: {len(res):,} rows (Source 1 has {len(src[1]):,}), {n_match:,} matched ids, "
              f"{sing:,} empty rows ({sing / max(len(res), 1):.1%})")
    if cand is not None:
        if cand.keys() != src[1]:
            errs.append(f"candidate_pairs: rows do not cover exactly the Source-1 ids "
                        f"({len(src[1] - cand.keys()):,} missing, {len(cand.keys() - src[1]):,} extra)")
        n_c = sum(len(v) for v in cand.values())
        print(f"candidate_pairs.tsv: {len(cand):,} rows, {n_c:,} candidate pairs ({n_c / max(len(cand), 1):.1f} per S1)")
        if res is not None:
            notcand = sum(len(set(v) - set(cand.get(k, ()))) for k, v in res.items() if v)
            if notcand:
                errs.append(f"{notcand:,} matched pairs are not in candidate_pairs.tsv")
    for x in errs[:30]:
        print("ERROR:", x)
    if len(errs) > 30:
        print(f"... and {len(errs) - 30:,} more errors")
    print("OK: no format errors" if not errs else f"FAILED: {len(errs):,} errors")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
