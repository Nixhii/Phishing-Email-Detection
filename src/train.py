import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib

MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_model(X_train, y_train, model_type="random_forest", **kwargs):
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 10),
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(
            C=kwargs.get("C", 1.0),
            max_iter=kwargs.get("max_iter", 1000),
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
    elif model_type == "svm":
        model = SVC(
            C=kwargs.get("C", 1.0),
            kernel=kwargs.get("kernel", "rbf"),
            probability=True,
            random_state=42,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from: random_forest, logistic_regression, svm")
    model.fit(X_train, y_train)
    return model


def save_model(model, model_name="phishing_model.joblib"):
    filepath = os.path.join(MODELS_PATH, model_name)
    joblib.dump(model, filepath)
    print(f"[INFO] Model saved to {filepath}")
    return filepath


def load_model(model_name="phishing_model.joblib"):
    filepath = os.path.join(MODELS_PATH, model_name)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model found at {filepath}")
    return joblib.load(filepath)
