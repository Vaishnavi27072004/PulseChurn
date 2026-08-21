"""Simple Streamlit fallback dashboard for Employee ID predictions."""

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/api/predict"

st.set_page_config(page_title="Customer Churn Prediction", page_icon="CC", layout="centered")

st.title("Employee Attrition Prediction")
st.caption("Predict and explain customer churn risk using telecom data.")
st.divider()

st.subheader("Look up a customer")
st.write("Enter a Customer ID from the `customerID` column in the Telco CSV file.")
employee_id = st.text_input("Customer ID", placeholder="Example: 5575-GNVDE")

if st.button("Predict", type="primary", use_container_width=True):
    if not employee_id.strip():
        st.error("Please enter an Employee ID.")
    else:
        try:
            response = requests.post(
                API_URL,
                json={"employeeId": employee_id.strip()},
                timeout=15,
            )
            result = response.json()
            if response.status_code == 404:
                st.error("Customer ID not found. Check the ID and try again.")
            elif response.status_code == 503:
                st.error("The model is not ready. Run the Telco data preparation and training steps first.")
            elif response.status_code == 422:
                st.error("This customer record is missing information needed for prediction.")
            elif response.status_code != 200:
                st.error(result.get("detail", "Prediction service returned an error."))
            else:
                is_high_risk = result["riskLevel"] == "High"
                if is_high_risk:
                    st.error("High Churn Risk")
                else:
                    st.success("Low Churn Risk")

                st.write(result["message"])
                st.metric("Confidence", f"{result['confidence']:.1%}")

                st.subheader("Customer information")
                employee = result["employee"]
                details = {
                    "Customer ID": result["employeeId"],
                    "Internet service": employee["department"],
                    "Contract": employee["jobRole"],
                    "Tenure status": employee["jobLevel"],
                    "Tenure (years)": employee["yearsAtCompany"],
                    "Monthly charges": f"${employee['monthlyIncome']}",
                    "Online security": employee["jobSatisfaction"],
                    "Payment method": employee["overtime"],
                    "Partner": employee["workLifeBalance"],
                }
                st.dataframe(details.items(), hide_index=True, use_container_width=True)
                st.subheader("What is driving this score?")
                st.dataframe(result["explanations"], hide_index=True, use_container_width=True)
        except requests.exceptions.RequestException:
            st.error("The prediction service is unavailable. Start `4_backend_api.py` first.")

st.divider()
st.caption("Customer Churn Prediction | FastAPI decision support")
