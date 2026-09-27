"""config.py — every path and tunable number in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # project root (folder that holds code/, data/, output/)
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

# file name patterns -> data/data_train/train_source1.tsv, data/data_test/test_source1.tsv ...
# (a flat layout, data/train_source1.tsv, is also accepted; see data_path)
SOURCE_FILE = "{split}_source{n}.tsv"
GROUND_TRUTH_FILE = "train_ground_truth.tsv"


def data_path(name: str, split: str = "train") -> Path:
    """Locate a data file: DATA_DIR/data_<split>/<name>, falling back to DATA_DIR/<name>."""
    nested = DATA_DIR / f"data_{split}" / name
    return nested if nested.exists() or not (DATA_DIR / name).exists() else DATA_DIR / name

# ---------------------------------------------------------------- blocking
MAX_BLOCK_SIZE = 1000        # drop keys shared by more S2/S3 records than this (e.g. 'n:services')
TOP_K_NAME = 20              # per S1 entity and source: keep the best TOP_K_NAME by name-key score,
TOP_K_ADDR = 20              #   the best TOP_K_ADDR by address-key score,
TOP_K_TOTAL = 30             #   and the best TOP_K_TOTAL by total score (union of the three)
KEY_WEIGHT_POWER = 1.0
CHUNK_S1 = 100_000          # Source-1 rows per pipeline chunk (blocking + features + stage-1 scoring)
PAIR_BUDGET = 15_000_000     # max raw key-matches per blocking chunk; lower it if you run out of RAM

# ---------------------------------------------------------------- matching (filled in by tune.py)
PRUNE_P1 = 0.02              # ml: pairs with stage-1 score below this never become matches (and are
                             # dropped early in streamed runs); stage 2 is trained on the rest only
HGB_PARAMS = dict(max_iter=1000, learning_rate=0.05, max_leaf_nodes=127, l2_regularization=1.0, random_state=0)
COLLECT_P1 = 0.005           # --collect: keep every pair with stage-1 score >= this ...
COLLECT_RATE = 0.10          # ... plus this share of the other (easy) negatives, weighted 1/COLLECT_RATE
MATCH_THRESHOLD = 0.72       # combined score needed to call a pair a match
NAME_ONLY_THRESHOLD = 0.90   # name score needed when either address is missing
RELATIVE_MARGIN = 0.20       # per S1, drop candidates scoring this far below its best candidate

WORKERS = -1                 # rapidfuzz threads (-1 = all cores)
FEATURE_PROCS = 0            # processes for the per-pair feature loop (0 = min(12, cores - 2))
