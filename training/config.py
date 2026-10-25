from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = Path(__file__).resolve().parent
DATA_DIR = TRAINING_DIR / "data"
MODELS_DIR = TRAINING_DIR / "models"
RAW_DATA_FILE = DATA_DIR / "raw_data.csv"
LOGS_DIR = TRAINING_DIR / "logs"
EXPERIMENTS_DIR = REPOSITORY_ROOT / "experiments"
EXPERIMENT_RUNS_DIR = EXPERIMENTS_DIR / "runs"
MODEL_NAME = "generalized_momentum"
LOG_EVERY_N_ROUNDS = 50

FORWARD_BARS = 8
TARGET_THRESHOLD = 0.005
TEST_SIZE = 0.2
COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.002
MIN_CONFIDENCE = 0.65

WALK_FORWARD_MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.01,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "class_weight": "balanced",
    "verbose": -1,
    "random_state": 42,
}
FINAL_MODEL_PARAMS = {**WALK_FORWARD_MODEL_PARAMS, "n_estimators": 500}
SMOKE_MODEL_PARAMS = {**FINAL_MODEL_PARAMS, "n_estimators": 30}
SMOKE_WALK_FORWARD_PARAMS = {**WALK_FORWARD_MODEL_PARAMS, "n_estimators": 20}
