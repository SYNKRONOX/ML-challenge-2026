# Context for Claude Code

Entity-resolution hackathon. For every Source 1 entity, list matching Source 2 / Source 3 ids.
Scored on macro F0.5 per Source 1 entity (false merges cost ~2x a miss; empty list for singletons).
No external data, APIs or geocoding allowed. Only the provided TSVs.

Data (tab-separated, in data/): train_source{1,2,3}.tsv, train_ground_truth.tsv, test_source{1,2,3}.tsv
Columns: entity_id, business_name, business_address, country (US / India).
Facts: 2.2M S1, ~5M S2, ~5.3M S3. Only 5.6% of S1 are singletons. Each S2/S3 id belongs to at most one S1.
Noise: accents, leetspeak, website names, Indic-script names, random replacement names, shuffled addresses,
state names in 3 forms, missing numbers, ~4% empty addresses.

Data lives in data/data_train/ and data/data_test/ (config.data_path resolves it). Use the project venv:
.venv/Scripts/python.exe (python.org 3.12; the MSYS2 python on PATH has no pip).

Pipeline (code/): normalize (+ learned Indic dict output/translit_dict.json, cache output/cache/) -> blocking
(BlockIndex per S2/S3, S1 streamed in 100k chunks) -> features (48) -> 2-stage stacked GBDT (--matcher ml)
-> decide (threshold, one-owner, relative margin). Commands: see run_pipeline.py docstring / Documentation_template.md.
Status (2026-09-27): DONE. Final model trained on 1/4 of S1 via --collect; OOF macro F0.5 0.9679 on held-out
sample-20 entities (blocking recall 0.970, P 0.991, R 0.932, singleton acc 0.961). Test run 63 min;
submission.zip built. Experiment log: output/logs/experiments.md. Format checks: code/check_submission.py
(utils/validate_submission.py was never provided). Always test on --sample 20 before full runs.
