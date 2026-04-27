import numpy as np

from .feature_engineering import FeatureEngineer
from .train import load_model


def predict(X, feature_engineer=None, model=None):
    if feature_engineer is None:
        feature_engineer = FeatureEngineer.load()

    if model is None:
        model = load_model()

    X_transformed = feature_engineer.transform(X)
    predictions = model.predict(X_transformed)
    return predictions


def predict_with_confidence(X, feature_engineer=None, model=None):
    if feature_engineer is None:
        feature_engineer = FeatureEngineer.load()

    if model is None:
        model = load_model()

    X_transformed = feature_engineer.transform(X)
    predictions = model.predict(X_transformed)
    probabilities = model.predict_proba(X_transformed)
    confidence = np.max(probabilities, axis=1)
    return predictions, confidence
