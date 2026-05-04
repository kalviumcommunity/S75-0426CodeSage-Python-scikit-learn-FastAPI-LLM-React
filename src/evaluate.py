"""
evaluate.py

Responsible for:
- Evaluating trained models on test data
- Generating performance metrics (accuracy, precision, recall, etc.)
- Saving evaluation reports to reports/
"""

from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and returns key metrics.

    Parameters:
        model: Trained model object
        X_test: Test feature matrix
        y_test: Test target vector

    Returns:
        accuracy: Accuracy score
        report: Detailed classification report
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def save_evaluation_report(report, path):
    """
    Placeholder for saving the evaluation report to a file.

    Parameters:
        report: Evaluation metrics or report string
        path: Destination file path
    """
    pass
