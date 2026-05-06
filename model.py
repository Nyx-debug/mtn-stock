import os
import joblib
from sklearn.ensemble import RandomForestRegressor


def train_model(X_train, y_train, model_path="models/mtn_model.pkl"):
    """
    Train a Random Forest regressor and save it to disk.
    """
    print("[Model] Training Random Forest...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    print(f"[Model] ? Model saved to {model_path}")
    return model


def load_model(model_path="models/mtn_model.pkl"):
    """
    Load the trained model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found. Train first: {model_path}")

    print(f"[Model] Loading model from {model_path}...")
    return joblib.load(model_path)


def predict(model, X_test):
    """
    Make predictions using a trained model.
    """
    print("[Model] Predicting...")
    return model.predict(X_test)
