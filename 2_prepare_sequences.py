"""Prepare the Telco customer churn dataset for model training."""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
RANDOM_STATE = 42

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def main() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    required_columns = set(FEATURES + ["customerID", "Churn"])
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            "The Telco dataset is missing required columns: " + ", ".join(missing_columns)
        )
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])
    df["Churn"] = (df["Churn"] == "Yes").astype(np.int32)

    duplicate_count = int(df.duplicated(subset=FEATURES).sum())
    if duplicate_count:
        df = df.drop_duplicates(subset=FEATURES, keep="first").reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=RANDOM_STATE, stratify=df["Churn"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_df["Churn"]
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df[FEATURES]).astype("float32")
    X_val = preprocessor.transform(val_df[FEATURES]).astype("float32")
    X_test = preprocessor.transform(test_df[FEATURES]).astype("float32")
    joblib.dump(preprocessor, f"{MODEL_DIR}/preprocessor.joblib")

    arrays = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": train_df["Churn"].to_numpy(),
        "y_val": val_df["Churn"].to_numpy(),
        "y_test": test_df["Churn"].to_numpy(),
    }
    for name, values in arrays.items():
        np.save(f"{PROCESSED_DIR}/{name}.npy", values)

    metadata = {
        "dataset": "Telco Customer Churn",
        "id_column": "customerID",
        "target": "Churn",
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "input_dim": int(X_train.shape[1]),
        "n_train": int(len(train_df)),
        "n_validation": int(len(val_df)),
        "n_test": int(len(test_df)),
        "duplicate_rows_removed": duplicate_count,
        "split": "70/15/15 stratified by Churn, random_state=42",
    }
    with open(f"{MODEL_DIR}/model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(json.dumps(metadata, indent=2))
    print("Preprocessing was fitted on the training split only.")


if __name__ == "__main__":
    main()