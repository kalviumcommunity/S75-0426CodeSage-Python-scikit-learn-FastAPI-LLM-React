from data_loader import load_data
from preprocessing import preprocess_data
from model import train_model
from evaluate import evaluate_model


def main():
    print("=== ML Workflow Starting ===")
    
    print("1. Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    print("2. Preprocessing data...")
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
    
    print("3. Training model...")
    model = train_model(X_train_scaled, y_train)
    
    print("4. Evaluating model...")
    accuracy, report = evaluate_model(model, X_test_scaled, y_test)
    
    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    print("=== Workflow Complete ===")


if __name__ == "__main__":
    main()
