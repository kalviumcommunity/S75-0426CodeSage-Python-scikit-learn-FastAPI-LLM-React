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


if __name__ == "__main__":
    # Sample prediction for demonstration
    sample_data = np.array([
        [5.1, 3.5, 1.4, 0.2],  # Setosa
        [6.7, 3.0, 5.2, 2.3]   # Virginica
    ])
    
    print("=== Running Sample Prediction ===")
    preds, confs = predict_with_confidence(sample_data)
    
    classes = ['Setosa', 'Versicolor', 'Virginica']
    for i, (p, c) in enumerate(zip(preds, confs)):
        print(f"Sample {i+1}: Predicted={classes[p]} (Confidence={c:.4f})")
