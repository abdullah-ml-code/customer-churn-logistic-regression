from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(
    "customer_churn_logistic_regression.joblib"
)


METRICS_PATH = Path(
    "artifacts/customer_churn_metrics.json"
)


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with METRICS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


if not MODEL_PATH.exists():
    st.error(
        "Model file was not found."
    )
    st.stop()


st.title(
    "Customer Churn Prediction"
)

st.write(
    "Predict customer churn probability using Logistic Regression."
)


pipeline = load_model()


if METRICS_PATH.exists():
    metrics = load_metrics()

    st.subheader(
        "Model Performance"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Accuracy",
            f"{metrics['accuracy']:.3f}",
        )

    with col2:
        st.metric(
            "Precision",
            f"{metrics['precision']:.3f}",
        )

    with col3:
        st.metric(
            "Recall",
            f"{metrics['recall']:.3f}",
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{metrics['f1_score']:.3f}",
        )

    with col5:
        st.metric(
            "ROC-AUC",
            f"{metrics['roc_auc']:.3f}",
        )


st.divider()

st.subheader(
    "Customer Information"
)


with st.form(
    "customer_form"
):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"],
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"],
        )

        tenure = st.number_input(
            "Tenure",
            min_value=0,
            max_value=100,
            value=12,
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
        )

    with col2:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No phone service",
                "No",
                "Yes",
            ],
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No",
            ],
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "No internet service",
                "No",
                "Yes",
            ],
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No internet service",
                "No",
                "Yes",
            ],
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "No internet service",
                "No",
                "Yes",
            ],
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "No internet service",
                "No",
                "Yes",
            ],
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No internet service",
                "No",
                "Yes",
            ],
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No internet service",
                "No",
                "Yes",
            ],
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year",
            ],
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=800.0,
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        use_container_width=True,
    )


if submitted:

    customer = pd.DataFrame(
        {
            "gender": [gender],
            "SeniorCitizen": [senior_citizen],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
        }
    )

    probabilities = pipeline.predict_proba(
        customer
    )[0]

    classes = pipeline.named_steps[
        "model"
    ].classes_

    yes_index = list(
        classes
    ).index("Yes")

    churn_probability = float(
        probabilities[yes_index]
    )

    prediction = (
        "Churn"
        if churn_probability >= 0.50
        else "No Churn"
    )

    st.divider()

    st.subheader(
        "Prediction Result"
    )

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Prediction",
            prediction,
        )

    with result_col2:
        st.metric(
            "Churn Probability",
            f"{churn_probability:.2%}",
        )

    st.progress(
        churn_probability
    )
