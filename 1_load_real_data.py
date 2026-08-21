"""Validate the Telco customer churn CSV used by the pipeline."""

import os

import pandas as pd

DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
REQUIRED_COLUMNS = {"customerID", "Churn", "InternetService", "Contract"}


def main() -> None:
    os.makedirs("data/raw", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Telco dataset not found at {DATA_PATH}.")

    data = pd.read_csv(DATA_PATH)
    missing_columns = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing_columns:
        raise ValueError("HR dataset is missing columns: " + ", ".join(missing_columns))

    print(f"Loaded {len(data)} customer records from {DATA_PATH}")
    print(f"Churn rate: {(data['Churn'] == 'Yes').mean():.1%}")
    print("Telco data validation complete. Next: run 2_prepare_sequences.py")


if __name__ == "__main__":
    main()