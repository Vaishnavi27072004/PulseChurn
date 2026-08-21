"""FastAPI service for Telco customer churn predictions."""

from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"

app = FastAPI(title="Employee Churn Prediction API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    employee_data = pd.read_csv(DATA_PATH)
    employee_data["TotalCharges"] = pd.to_numeric(employee_data["TotalCharges"], errors="coerce")
    employee_data["TotalCharges"] = employee_data["TotalCharges"].fillna(
        employee_data["tenure"] * employee_data["MonthlyCharges"]
    )
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
except (FileNotFoundError, OSError, ValueError) as error:
    employee_data = pd.DataFrame()
    preprocessor = None
    model = None
    print(f"Model assets are not ready: {error}")

class PredictionRequest(BaseModel):
    employeeId: str = Field(min_length=1, max_length=64)


class PredictionResponse(BaseModel):
    employeeId: str
    prediction: str
    probability: float
    confidence: float
    riskLevel: str
    message: str
    employee: dict
    explanations: list[dict]
    timestamp: str


def find_employee(employee_id: str) -> pd.Series:
    normalized_id = employee_id.strip().casefold()
    if employee_data.empty:
        raise HTTPException(status_code=503, detail="Model assets are not ready. Run the training pipeline first.")
    match = employee_data[employee_data["customerID"].astype(str).str.casefold() == normalized_id]
    if match.empty:
        raise HTTPException(
            status_code=404,
            detail="Customer ID not found. Use a valid ID such as 5575-GNVDE.",
        )
    return match.iloc[0]


def explain_prediction(transformed: np.ndarray) -> list[dict]:
    feature_names = list(preprocessor.get_feature_names_out())
    feature_values = transformed[0]
    contributions = np.asarray(model.coef_[0]) * feature_values
    active_features = []

    for index, feature_name in enumerate(feature_names):
        is_categorical = feature_name.startswith("categorical__")
        if is_categorical and feature_values[index] == 0:
            continue
        if contributions[index] == 0:
            continue
        active_features.append(index)

    ranked = sorted(active_features, key=lambda index: abs(contributions[index]), reverse=True)
    return [
        {
            "feature": feature_names[index].split("__", 1)[-1].replace("_", " "),
            "impact": round(float(contributions[index]), 4),
            "direction": "Increases churn risk" if contributions[index] > 0 else "Reduces churn risk",
        }
        for index in ranked[:5]
    ]


@app.get("/health")
def health():
    return {"status": "ok", "employees": int(len(employee_data)), "model_ready": model is not None}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    row = find_employee(request.employeeId)
    if preprocessor is None or model is None:
        raise HTTPException(status_code=503, detail="Model assets are not ready. Run the training pipeline first.")
    feature_frame = pd.DataFrame([row])
    try:
        transformed = preprocessor.transform(feature_frame).astype("float32")
        churn_probability = float(model.predict_proba(transformed)[0, 1])
        explanations = explain_prediction(transformed)
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Employee information is incomplete: {error}") from error

    risk_level = "High" if churn_probability >= 0.5 else "Low"
    prediction = "Churn" if risk_level == "High" else "Stay"
    return PredictionResponse(
        employeeId=str(row["customerID"]),
        prediction=prediction,
        probability=round(churn_probability, 4),
        confidence=round(churn_probability if prediction == "Churn" else 1 - churn_probability, 4),
        riskLevel=risk_level,
        message=(
            "This employee is predicted to churn."
            if prediction == "Churn"
            else "This employee is predicted to stay."
        ),
        employee={
            "department": str(row["InternetService"]), "jobRole": str(row["Contract"]),
            "jobLevel": "New tenure" if int(row["tenure"]) < 12 else "Established",
            "yearsAtCompany": round(float(row["tenure"]) / 12, 1),
            "monthlyIncome": round(float(row["MonthlyCharges"]), 2),
            "jobSatisfaction": str(row["OnlineSecurity"]), "overtime": str(row["PaymentMethod"]),
            "workLifeBalance": str(row["Partner"]),
        },
        explanations=explanations,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)