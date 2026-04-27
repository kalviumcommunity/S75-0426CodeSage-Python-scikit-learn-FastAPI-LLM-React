from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"

SCALER_PATH = MODELS_DIR / "scaler.pkl"
MODEL_PATH = MODELS_DIR / "model.pkl"
