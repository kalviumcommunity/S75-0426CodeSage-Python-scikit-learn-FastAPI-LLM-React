"""
predict.py

Responsible for:
- Standalone inference using saved artifacts
- Loading the trained model and feature engineer
- Validating and transforming new input data
- Generating predictions and confidence scores
"""

import joblib
import argparse
import pandas as pd
import numpy as np
from .config import MODEL_PATH, SCALER_PATH

def load_artifacts():
    """
    Loads the saved model and feature engineer artifacts.
    """
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(
            "Model or scaler artifacts not found. Please run src/train.py first."
        )
    
    model = joblib.load(MODEL_PATH)
    feature_engineer = joblib.load(SCALER_PATH)
    return model, feature_engineer

def predict(input_data: pd.DataFrame):
    """
    Generates predictions for the provided input data.

    Parameters:
        input_data: DataFrame containing features

    Returns:
        dict: Predictions and confidence scores
    """
    model, feature_engineer = load_artifacts()
    
    # Transform data (using transform, NOT fit_transform)
    X_transformed = feature_engineer.transform(input_data)
    
    # Generate predictions
    predictions = model.predict(X_transformed)
    probabilities = model.predict_proba(X_transformed)
    confidence = np.max(probabilities, axis=1)
    
    return {
        "predictions": predictions.tolist(),
        "confidence": confidence.tolist()
    }

def main():
    parser = argparse.ArgumentParser(description="Run inference on new data.")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the input CSV file for prediction"
    )
    args = parser.parse_args()
    
    try:
        print(f"Loading input data from {args.input}...")
        input_df = pd.read_csv(args.input)
        
        # Basic validation: ensure we don't include the target column if it's there
        if 'target' in input_df.columns:
            input_df = input_df.drop(columns=['target'])
        
        print("Generating predictions...")
        results = predict(input_df)
        
        classes = ['Setosa', 'Versicolor', 'Virginica']
        print("\n=== Prediction Results ===")
        for i, (p, c) in enumerate(zip(results['predictions'], results['confidence'])):
            class_name = classes[p] if p < len(classes) else f"Class {p}"
            print(f"Row {i+1}: Predicted={class_name} (Confidence={c:.4f})")
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
