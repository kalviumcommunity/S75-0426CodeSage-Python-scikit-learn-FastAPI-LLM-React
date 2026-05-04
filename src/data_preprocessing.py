"""
data_preprocessing.py

Responsible for:
- Loading raw data from data/raw/
- Cleaning and handling missing values
- Splitting data into training and testing sets
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from .config import RANDOM_STATE, TEST_SIZE

def load_data(test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Loads the raw dataset and splits it into train and test sets.

    Parameters:
        test_size: Proportion of the dataset to include in the test split
        random_state: Seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def clean_data(data):
    """
    Placeholder for data cleaning logic.

    Parameters:
        data: Raw data frame or matrix

    Returns:
        Cleaned data
    """
    pass
