from .data_preprocessing import load_data
from .feature_engineering import FeatureEngineer
from .train import run_training_pipeline, load_model
from .evaluate import evaluate_model


def main():
    print("=== ML Workflow Starting ===")

    print("1. Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    print("2. Feature engineering and training...")
    model, feature_engineer = run_training_pipeline(X_train, y_train)

    print("3. Evaluating model...")
    X_test_transformed = feature_engineer.transform(X_test)
    accuracy, report = evaluate_model(model, X_test_transformed, y_test)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    print("=== Workflow Complete ===")


if __name__ == "__main__":
    main()
