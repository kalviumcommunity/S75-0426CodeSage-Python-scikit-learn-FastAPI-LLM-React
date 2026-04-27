import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from .config import MODEL_PATH, N_ESTIMATORS, RANDOM_STATE
from .feature_engineering import FeatureEngineer


def train_model(X_train, y_train, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE):
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, path=None):
    path = Path(path) if path else MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path=None):
    path = Path(path) if path else MODEL_PATH
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def run_training_pipeline(X_train, y_train, feature_engineer=None, save_artifacts=True):
    if feature_engineer is None:
        feature_engineer = FeatureEngineer()

    X_train_transformed = feature_engineer.fit_transform(X_train)

    model = train_model(X_train_transformed, y_train)

    if save_artifacts:
        feature_engineer.save()
        save_model(model)

    return model, feature_engineer
