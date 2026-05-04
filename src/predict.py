"""
predict.py

Responsible for:
- Loading trained model and transformation artifacts
- Processing new data for prediction
- Generating predictions and confidence scores
"""

import numpy as np
from .feature_engineering import FeatureEngineer
from .train import load_model

def predict(X, feature_engineer=None, model=None):
    """
    Generates predictions for new data.

    Parameters:
        X: Input features
        feature_engineer: Loaded FeatureEngineer instance
        model: Loaded model object

    Returns:
        predictions: Array of predicted classes
    """
    if feature_engineer is None:
        feature_engineer = FeatureEngineer.load()

    if model is None:
        model = load_model()

    X_transformed = feature_engineer.transform(X)
    predictions = model.predict(X_transformed)
    return predictions

def predict_with_confidence(X, feature_engineer=None, model=None):
    """
    Generates predictions and confidence scores for new data.

    Parameters:
        X: Input features
        feature_engineer: Loaded FeatureEngineer instance
        model: Loaded model object

    Returns:
        predictions: Array of predicted classes
        confidence: Array of confidence scores
    """
    if feature_engineer is None:
        feature_engineer = FeatureEngineer.load()

    if model is None:
        model = load_model()

    X_transformed = feature_engineer.transform(X)
    predictions = model.predict(X_transformed)
    probabilities = model.predict_proba(X_transformed)
    confidence = np.max(probabilities, axis=1)
    return predictions, confidence

def format_prediction_output(predictions, confidence):
    """
    Placeholder for formatting the prediction output for an API or UI.

    Parameters:
        predictions: Array of predictions
        confidence: Array of confidence scores

    Returns:
        Formatted output (e.g., JSON)
    """
    pass

if __name__ == "__main__":
    # Sample prediction for demonstration
    sample_data = np.array([
        [5.1, 3.5, 1.4, 0.2],  # Setosa
        [6.7, 3.0, 5.2, 2.3]   # Virginica
    ])
    
    print("=== Running Sample Prediction ===")
    try:
        preds, confs = predict_with_confidence(sample_data)
        classes = ['Setosa', 'Versicolor', 'Virginica']
        for i, (p, c) in enumerate(zip(preds, confs)):
            print(f"Sample {i+1}: Predicted={classes[p]} (Confidence={c:.4f})")
    except FileNotFoundError:
        print("Model or scaler not found. Please run main.py first to train the model.")
