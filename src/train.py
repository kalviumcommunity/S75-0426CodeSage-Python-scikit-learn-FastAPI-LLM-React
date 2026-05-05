"""
train.py

Responsible for:
- Orchestrating the training pipeline
- Loading data using data_loader
- Splitting data into train and test sets
- Fitting preprocessors on training data only
- Training the ML model
- Evaluating on test data
- Saving all artifacts (model, scaler, report)
"""

import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from .data_loader import load_data
from .feature_engineering import FeatureEngineer
from .evaluate import evaluate_model
from .config import (
    TRAIN_DATA_PATH, 
    MODEL_PATH, 
    SCALER_PATH, 
    REPORT_PATH, 
    TEST_SIZE, 
    RANDOM_STATE,
    N_ESTIMATORS
)

def build_model(X_train, y_train, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE):
    """
    Trains the primary ML model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def run_training():
    """
    Executes the full training workflow.
    """
    print(f"Loading data from {TRAIN_DATA_PATH}...")
    df = load_data(TRAIN_DATA_PATH)
    
    # Assuming the last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print("Fitting feature engineer on training data...")
    feature_engineer = FeatureEngineer()
    X_train_transformed = feature_engineer.fit_transform(X_train)
    
    print("Training model...")
    model = build_model(X_train_transformed, y_train)
    
    print("Evaluating on test set...")
    X_test_transformed = feature_engineer.transform(X_test)
    accuracy, report = evaluate_model(model, X_test_transformed, y_test)
    
    print(f"Accuracy: {accuracy:.4f}")
    
    print("Saving artifacts...")
    # Save model and scaler using joblib
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_engineer, SCALER_PATH)
    
    # Save evaluation report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"Artifacts saved to {MODEL_PATH}, {SCALER_PATH}, and {REPORT_PATH}")
    print("Training complete.")

if __name__ == "__main__":
    run_training()
