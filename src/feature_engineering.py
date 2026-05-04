"""
feature_engineering.py

Responsible for:
- Transforming raw features into model-ready inputs
- Scaling, encoding, and dimensionality reduction
- Saving and loading transformation artifacts
"""

import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from .config import SCALER_PATH

class FeatureEngineer:
    """
    Handles feature transformation and scaling.
    """
    def __init__(self, scaler=None):
        self.scaler = scaler if scaler is not None else StandardScaler()

    def fit(self, X_train):
        """Fits the transformer on training data."""
        self.scaler.fit(X_train)
        return self

    def transform(self, X):
        """Applies transformation to new data."""
        return self.scaler.transform(X)

    def fit_transform(self, X_train):
        """Fits and transforms in one step."""
        self.fit(X_train)
        return self.transform(X_train)

    def save(self, path=None):
        """Saves the transformation artifact."""
        path = Path(path) if path else SCALER_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path=None):
        """Loads a saved transformation artifact."""
        path = Path(path) if path else SCALER_PATH
        with open(path, 'rb') as f:
            scaler = pickle.load(f)
        return cls(scaler=scaler)

def extract_features(data):
    """
    Placeholder for custom feature extraction logic.

    Parameters:
        data: Preprocessed data

    Returns:
        Extracted features
    """
    pass
