from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TEST_SIZE


def load_data(test_size=TEST_SIZE, random_state=RANDOM_STATE):
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
