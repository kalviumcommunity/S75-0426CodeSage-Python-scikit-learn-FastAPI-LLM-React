"""
main.py

Responsible for:
- Orchestrating the full ML pipeline
- Connecting data loading, feature engineering, training, and evaluation
- Providing a single entry point for the project
"""

from .data_preprocessing import load_data
from .feature_engineering import FeatureEngineer
from .train import train_model, save_model
from .evaluate import evaluate_model

def run_workflow():
    """
    Orchestrates the end-to-end machine learning workflow.
    """
    print("=== ML Workflow Starting ===")

    print("1. Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    print("2. Feature engineering and training...")
    feature_engineer = FeatureEngineer()
    X_train_transformed = feature_engineer.fit_transform(X_train)
    model = train_model(X_train_transformed, y_train)

    print("3. Evaluating model...")
    X_test_transformed = feature_engineer.transform(X_test)
    accuracy, report = evaluate_model(model, X_test_transformed, y_test)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    print("4. Saving artifacts...")
    feature_engineer.save()
    save_model(model)

    print("=== Workflow Complete ===")

def setup_environment():
    """
    Placeholder for environment setup or verification logic.
    """
    pass

if __name__ == "__main__":
    run_workflow()
