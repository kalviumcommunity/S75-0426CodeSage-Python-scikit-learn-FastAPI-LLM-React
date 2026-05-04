"""
config.py

Responsible for:
- Centralizing all configuration parameters
- Managing project-wide file paths
- Ensuring consistency across modules
"""

from pathlib import Path

# General Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100

# Base Directory
BASE_DIR = Path(__file__).parent.parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Artifact Directories
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# File Paths
SCALER_PATH = MODELS_DIR / "scaler.pkl"
MODEL_PATH = MODELS_DIR / "model.pkl"
TRAIN_DATA_PATH = RAW_DATA_DIR / "iris.csv"  # Placeholder for raw data
