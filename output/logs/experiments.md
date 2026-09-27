# Experiment log

All numbers are on the train split with `--sample 20` (1/20 of Source-1, crc32 hash sample; all of S2/S3)
unless stated otherwise. Rules F0.5 = coordinate-ascent tuned on the sample; ML F0.5 = 2-fold
out-of-fold (split by Source-1 entity) with tuned decision thresholds.

| # | change | block recall | cand pairs | rules F0.5 | ml F0.5 (OOF) | kept? |
|---|--------|-------------:|-----------:|-----------:|--------------:|-------|
| 0 | baseline (untuned rules 0.7892; tuned below) | 0.9291 | 8.14M | 0.8291 | 0.9169 | baseline |
| 1 | blocking: name token-pair keys, full-name key, number-pair keys; size filter on S2/S3 side only; per-family top-K (name 20 / addr 20 / total 30) | 0.9649 | 8.72M | – | – | yes (recall +0.036, pairs +7%) |
| 2 | learned Indic->English token dictionary (build_translit.py, from non-sample train entities); sree/shrii as honorifics | 0.9703 | 8.76M | – | – | yes (recall +0.005) |
| 3 | features: number containment/closeness/digit-sim, fuzzy token alignment + unmatched-token IDF, unseen-in-S1 share, domain containment/initials, name/address frequency; IDF from full S1; ML stacked 2-stage (competition features), stage-1 prune 0.02 | 0.9702 | 8.76M | 0.8325 (0.8343 w/ old params) | **0.9629** (stage-1 only 0.9604) | yes (+0.046) |
| 3b | decision rule: expected-F0.5 prefix per entity vs threshold+margin | – | – | – | 0.9628 vs 0.9629 | no (no gain; margin 0.4/0.5/1.0: 0.9621/0.9605/0.9605) |
| 3c | learning curve (stage-1, held-out fold): 1/8, 1/4, 1/2 of sample -> 0.9559, 0.9572, 0.9592; 1000 iters lr .05 leaves 127 -> 0.9598 | – | – | – | – | info: more data ~+0.002 per doubling |
| 4 | features: S1 name-count (both sides) and S1 address-count; blocking join via sorted keys + searchsorted (same pairs, ~30% faster run) | 0.9702 | 8.76M | 0.8326 | **0.9642** (stage-1 0.9621; singleton acc 0.9487) | yes (+0.0013 < 0.002 -> stop feature rounds) |
| 5 | PHASE 4: weighted training set from 1/4 of S1 (--collect: all true + stage-1 >= .005 + 10% easy negatives x10; 6.6M rows ~ 43.9M effective); GBDT 1000 it / lr .05 / 127 leaves; scored on held-out sample-20 entities | 0.9699 (on 1/4) | 44M | – | **0.9679** (stage-1 0.9649; P .9908 R .9322 singleton .9614) | yes (+0.0037) — FINAL MODEL |

Baseline detail (sample 20, 109,970 S1, 380,789 true pairs, 6,065 singletons):
- rules tuned: P 0.8878  R 0.7700  singleton acc 0.6030  F0.5 0.8291
- ml OOF:      P 0.9748  R 0.8416  singleton acc 0.8780  F0.5 0.9169
