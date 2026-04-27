import pickle
from pathlib import Path

from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE, SCALER_PATH


class FeatureEngineer:
    def __init__(self, scaler=None):
        self.scaler = scaler if scaler is not None else StandardScaler()

    def fit(self, X_train):
        self.scaler.fit(X_train)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

    def fit_transform(self, X_train):
        self.fit(X_train)
        return self.transform(X_train)

    def save(self, path=None):
        path = Path(path) if path else SCALER_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path=None):
        path = Path(path) if path else SCALER_PATH
        with open(path, 'rb') as f:
            scaler = pickle.load(f)
        return cls(scaler=scaler)


def create_feature_engineer(random_state=RANDOM_STATE):
    return FeatureEngineer(scaler=StandardScaler())
