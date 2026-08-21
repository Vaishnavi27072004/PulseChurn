"""Train a simple logistic-regression churn classifier."""

import json
import os

import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def metrics_for(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def main() -> None:
    X_train = np.load("data/processed/X_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_val = np.load("data/processed/y_val.npy")
    y_test = np.load("data/processed/y_test.npy")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    baseline = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")
    cv_auc = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="roc_auc")

    model = LogisticRegression(
        C=0.1,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    results = {
        "train": metrics_for(y_train, train_prob),
        "validation": metrics_for(y_val, val_prob),
        "test": metrics_for(y_test, test_prob),
        "cross_validation_train_auc_mean": float(cv_auc.mean()),
        "cross_validation_train_auc_std": float(cv_auc.std()),
        "model": "LogisticRegression",
        "hyperparameters": {"C": 0.1, "class_weight": "balanced", "max_iter": 1000},
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/churn_model.joblib")
    with open("models/model_metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()