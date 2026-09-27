# Entity Resolution — Methodology

**Task.** For every Source-1 (S1) business entity, list the Source-2 (S2) and Source-3 (S3) records that
describe the same business. Scoring is macro F0.5 per S1 entity: precision is weighted about twice as
heavily as recall, and an S1 entity with no true match must get an empty list.

**Constraints.** Only the provided TSV files are used: no external data, APIs, geocoding or
downloaded resources. Every learned artefact (the transliteration dictionary, the models and the
thresholds) is fitted on the **training split only**. The test split is used for nothing except the final run.

---

## 1. Pipeline overview

```
normalize ─► blocking (key index per S2/S3, S1 streamed in chunks) ─► pair features
         ─► stage-1 GBDT ─► prune (p1 < 0.02) ─► competition features ─► stage-2 GBDT
         ─► decision rules (threshold, one-owner, relative margin) ─► matching_results.tsv
```

| Step | Code |
|---|---|
| Normalization (+ cache) | `code/normalize.py`, `code/data_io.py` |
| Indic→English token dictionary | `code/build_translit.py` → `output/translit_dict.json` |
| Blocking | `code/blocking.py` |
| Features | `code/features.py` |
| Matchers and decision rules | `code/match.py` |
| Training and threshold tuning | `code/tune.py` |
| End-to-end run | `code/run_pipeline.py` |
| Error analysis | `code/analyze.py` |
| Metric | `code/evaluate.py` |
| Output format checks | `code/check_submission.py` |

### Commands to reproduce

```
pip install -r requirements.txt
python code/build_translit.py                                        # Indic->English dictionary (train only)
python code/run_pipeline.py --split train --evaluate --sample 20     # features for 1/20 of S1
python code/tune.py --matcher ml                                     # first model (used to pick hard negatives)
python code/run_pipeline.py --split train --sample 4 --collect       # weighted training set, 1/4 of S1
python code/tune.py --matcher ml --eval-sample 20                    # final model + thresholds
python code/run_pipeline.py --split test --matcher ml                # -> output/matching_results.tsv, candidate_pairs.tsv
python code/check_submission.py                                      # format checks
```

---

## 2. Normalization (`normalize.py`)

### Names
- **Scripts and accents:** any Indic-script token found in the learned dictionary (Section 3) is
  translated to English. Everything is then transliterated to ASCII (`unidecode`) and lower-cased.
- **Website names** (`www.x.com`, `caangel.com`) are reduced to their stem and flagged `is_domain`.
- **Glued-on numbers:** phone or registration numbers of 6 or more digits are removed.
- **Leetspeak:** digits inside alphabetic tokens are fixed (`h0rizon` → `horizon`).
- **Honorifics and legal words are dropped:**
  - honorifics: `shri`, `sree`, `smt`, `m/s`, `dr`, `the`, …
  - legal words: `pvt`, `ltd`, `llc`, `inc`, `corp`, `co`, `llp`, …
  - on transliterated names, legal words are also dropped by their phonetic skeleton (`praaivett`).
- **Duplicated words** are removed (`palghar palghar`).
- **Outputs:**
  - `core`: the space-joined core tokens;
  - `compact`: the same with no spaces;
  - `skel`: a phonetic skeleton;
  - flags `is_domain` and `is_translit`.

### Addresses
- Non-Latin tokens are dropped (in this data they are state names), as are state names and codes (all
  three forms: full name, 2-letter code, native script).
- Unit, PO box and house-number decorations are dropped (`unit`, `apt`, `po`, `box`, `no`, `h.no`, …).
- Street types and directions are abbreviated (`street` → `st`, `road` → `rd`, `north` → `n`, `nagar` → `ngr`, …).
- **Words and numbers are kept separately.** Words form a sorted set, since components are often
  shuffled. Numbers are split out of mixed tokens (`21/14` → {21, 14}, `1056c` → 1056), with leading zeros stripped.

### Caching
Normalized sources are cached in `output/cache/`. The cache key covers the raw file (size and mtime),
`normalize.py`, `data_io.py` and the dictionary, so any change invalidates it automatically.

## 3. Learned Indic→English token dictionary (`build_translit.py`)

About 750k S2/S3 names are written in an Indic script (Devanagari, Tamil, Telugu, Kannada, Gujarati,
Bengali, Gurmukhi, Malayalam, Odia). Generic transliteration does not line up with the English
spelling (`श्री टेक्नोलॉजी` → `shrii tteknolonjii` vs `Sree Technology`).

The vocabulary is small, though. From ground-truth pairs whose two names have the same number of
tokens, tokens are aligned by position. A mapping is kept when it was seen at least 2 times and
accounts for at least 60% of that token's alignments. The result maps 1,319 of 1,347 Indic tokens.

Only S1 entities **outside** the evaluation sample (`crc32(id) % 20 != 0`) are used, so the reported
scores stay honest. The dictionary is learned from training data only; the test split has no labels
and is never touched.

---

## 4. Blocking (`blocking.py`)

Every record emits cheap string keys, scoped by country. Two records become a candidate pair when
they share a key.

| Family | Key | Purpose |
|---|---|---|
| name | `n:<token>` | single name token (≥ 3 chars) |
| name | `p:<tok>:<tok>` | **pair** of name tokens — names come from a small vocabulary, so single tokens are often too common |
| name | `f:<compact>` | whole compact name (also covers 2-letter names like `om`, `rg`) |
| name | `k:<skeleton>` | phonetic skeleton token (transliterations) |
| name | `c:<prefix6>` | first 6 chars of the compact name (`caangel.com` ~ `CA Angel`) |
| address | `a:<num>:<word>` | house number + street/area word |
| address | `aa:<num>:<num>` | two numbers of the address |
| address | `w:<word>:<word>` | two address words (addresses without numbers) |

- **Oversized keys:** keys shared by more than `MAX_BLOCK_SIZE` = 1000 S2/S3 records are dropped. Only
  the S2/S3 side is counted, so a `--sample` run blocks exactly like the full run.
- **Scoring and top-K:** each pair gets a name-key score and an address-key score, each the sum of 1/size
  over the shared keys. For each S1 entity and source, the pipeline keeps the **union** of:
  - the top 20 by name score;
  - the top 20 by address score;
  - the top 30 by total score.

  A record that agrees only on the name (for example an empty address) is therefore not crowded
  out by the many other businesses at the same address, and vice versa.
- **Implementation:** each S2/S3 key table (about 75M rows) is built once and sorted. S1 chunks
  (100k rows) are joined against it with binary search (`searchsorted`), which keeps memory flat.

### Blocking recall (1/20 sample of train S1, 380,789 true pairs)

| Version | Recall | Pairs per S1 |
|---|---|---|
| Baseline (single tokens, top-40 by total score) | 0.9291 | 74.0 |
| + token-pair / full-name / number-pair keys, per-family top-K, S2/S3-side size filter | 0.9649 | 79.3 |
| + learned Indic→English dictionary | **0.9702** | 79.6 |

The remaining misses are mostly exact-name records with an empty address and a very common name,
with hundreds of namesakes. Even if blocking kept them, the matcher could not claim them safely.

---

## 5. Pair features (`features.py`, 48 features)

- **Name similarity** (`rapidfuzz`, vectorised): token-set, token-sort, compact ratio, partial ratio,
  and skeleton token-set ratio.
- **IDF-weighted overlaps** (IDF from all records of the split): name Jaccard and containment;
  address Jaccard and containment; total IDF of the shared address tokens; IDF of the rarest shared number.
- **House numbers:**
  - conflict flag;
  - share of each side's numbers found on the other side;
  - closest relative difference;
  - best digit-string similarity (`2204` ~ `204`, `131` ~ `31`).

  House numbers are noisy even on true matches.
- **Look-alike distractors:** fuzzy token alignment (ratio ≥ 0.75). Features: the number of unmatched
  tokens on each side, the IDF of the rarest unmatched token (a real replacement word such as `holdings`
  or `noodles`, as opposed to a typo), and the mean alignment score.
- **Random replacement names** (`Novidova`, `Lumfluxdrex`): share of the other name's tokens never
  seen in any S1 name.
- **Website names:** share of the other name's tokens contained in the domain, and whether the domain
  starts with their initials (`siprivate.com` ~ **S**hyam **I**nvestments **Private**).
- **Specificity:** log counts of records sharing exactly this name or address across all sources;
  log counts of **S1** entities with this exact name (either side's) or at this address. A record
  with no address is only claimable when its name is unique among S1 entities.
- **Blocking scores and flags:** name, address and total blocking score; address-missing,
  transliterated and domain flags.

## 6. Matcher (`match.py`, `tune.py`)

**Stacked gradient boosting** (`sklearn` `HistGradientBoostingClassifier`, 1000 iterations, learning
rate 0.05, 127 leaves, early stopping):

1. **Stage 1:** pair features → p1.
2. **Pruning:** pairs with p1 < 0.02 are dropped. This keeps about 5% of pairs and loses about 0.2% of true pairs.
3. **Stage 2:** pair features plus competition features computed from p1:
   - within the S1 entity: best p1, gap to it, rank, number of pairs with p1 > 0.5;
   - within the S2/S3 record: **best p1 of any other S1 entity** for the same record, gap to it,
     number of competing S1 entities.

   This is a learned version of the one-owner rule. It also lets the model reject distractors when a
   better-fitting S1 exists.

A hand-written rules scorer (`--matcher rules`) is kept as a baseline.

### Decision rules (both matchers)
1. Keep pairs with score ≥ `MATCH_THRESHOLD`.
2. **One-owner rule:** each S2/S3 record belongs to at most one S1 entity (true in the ground truth),
   so only its best-scoring S1 keeps it.
3. **Relative margin:** within an S1 entity, drop candidates scoring more than `RELATIVE_MARGIN` below
   its best one.

Singletons fall out naturally: an entity whose candidates all score below the threshold gets an empty list.

An alternative rule was also tested: choose per entity the top-k prefix (or nothing) that maximises
expected F0.5. It tied with the threshold rule (0.9628 vs 0.9629), so the simpler rule was kept.

### Tuning procedure (honest estimates)
- **Folds:** 2-fold cross-validation **by S1 entity** (crc32 hash). Stage-2 features are built from
  **out-of-fold** stage-1 scores, so stage 2 never sees optimistic p1 values.
- **Thresholds:** `MATCH_THRESHOLD` and `RELATIVE_MARGIN` are chosen by coordinate ascent on the
  out-of-fold macro F0.5.
- **Training set (`--collect`):**
  - built from 1/4 of train S1;
  - keeps every true pair, every pair the first model scores ≥ 0.005, and 10% of the remaining easy
    negatives with weight 10 (unbiased);
  - about 5× more entities than the 1/20 sample, since the learning curve showed about +0.002 F0.5 per
    doubling of the data;
  - scores are reported only on the 1/20-sample entities, which the dictionary never saw.
- **Final models:** refit on the whole collected set.
- **Streaming:** the full data has about 150M candidate pairs, so test (and full-train) runs are streamed.
  Each S1 chunk is scored by stage 1 immediately and only pairs with p1 ≥ 0.02 are kept, which is
  exactly the population stage 2 was trained on.

---

## 7. Results

Macro F0.5 is measured on train, 1/20 sample of S1 (109,970 entities, 6,065 singletons). ML scores
are out-of-fold; rules scores are tuned in-sample.

| # | Change | Blocking recall | Rules F0.5 | ML F0.5 |
|---|---|---|---|---|
| 0 | Baseline code, tuned | 0.9291 | 0.8291 | 0.9169 |
| 1–2 | New blocking keys, per-family top-K, Indic dictionary | 0.9702 | 0.8343 | – |
| 3 | New features + stacked 2-stage ML | 0.9702 | 0.8325 | 0.9629 |
| 4 | S1 name/address count features | 0.9702 | 0.8326 | 0.9642 |
| 5 | Final: 5× larger weighted training set, bigger GBDT | 0.9699 | – | **0.9679** |

Final model, out-of-fold on the held-out sample-20 entities (the ground truth has 380,789 true pairs for them):

| Metric | Value |
|---|---|
| Blocking recall | 0.970 |
| Mean precision | 0.9908 |
| Mean recall | 0.9322 |
| Singleton accuracy | 0.9614 |
| **Macro F0.5** | **0.9679** |

Stage 1 alone scores 0.9649, so stage 2 adds +0.003. The same model scored in-sample on these
entities gives 0.9732; that figure is optimistic and is not the estimate.

Test run (`--split test --matcher ml`, 63 min on 24 cores / 32 GB RAM machine):

| Item | Value |
|---|---|
| S1 entities | 1,732,544 (one row each in `matching_results.tsv`) |
| Candidate pairs | 140.4M (81 per entity) |
| Pairs passed to stage 2 | 7.64M |
| Matched ids | 5.74M (3.3 per entity) |
| Empty rows | 109,991 (6.3%) |

The test split behaved like train at every stage: pairs per entity (81 vs 80) and the share of pairs
kept by stage-1 pruning (about 5%). `code/check_submission.py` reports no format errors.

Full experiment log: `output/logs/experiments.md`.

## 8. Known limitations
- **Blocking ceiling of about 0.970.** Most misses are address-less records with very common names,
  and some transliterations aren't covered. Raising the ceiling would multiply the candidate count.
- **Address-less records with a name shared by several S1 entities** are genuinely ambiguous: 8–27% of
  them are true matches. The model declines them, which costs recall but protects precision.
- **Hard distractors** (the same name at the same street with a slightly changed house number, or one
  added word) remain the main source of false positives; house numbers are noisy on true matches too.
- **Memory.** The model was trained on 1/4 of the training entities rather than all of them. sklearn's
  boosting converts inputs to float64, and features for all ~175M training pairs would need more than
  60 GB of RAM.
- **Learned artefacts** (dictionary, models) assume the test split shares the training split's
  vocabulary and noise generator.
- **Output format.** No validator was provided. `candidate_pairs.tsv` uses the same layout as
  `matching_results.tsv` (`source1_entity_id<TAB>candidate_entity_ids`, comma-separated).
