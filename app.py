import streamlit as st
import pandas as pd
import numpy as np
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

artifact = joblib.load("models/churn_model.pkl")

model = artifact["model"]
scaler = artifact["scaler"]
ordinal_encoder = artifact["ordinal_encoder"]
feature_columns = artifact["feature_columns"]
continuous_cols = artifact["continuous_cols"]
threshold = artifact["threshold"]


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a customer is likely to churn based on "
    "their account and demographic information."
)


# --------------------------------------------------
# Customer Information
# --------------------------------------------------

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=10,
        value=5
    )

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    point_earned = st.number_input(
        "Point Earned",
        min_value=0,
        value=500,
        step=10
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    num_products = st.selectbox(
        "Number of Products",
        [1, 2, 3, 4]
    )

    has_cr_card = st.selectbox(
        "Has Credit Card?",
        ["Yes", "No"]
    )

    active_member = st.selectbox(
        "Is Active Member?",
        ["Yes", "No"]
    )

    satisfaction_score = st.slider(
        "Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3
    )

    card_type = st.selectbox(
        "Card Type",
        ["SILVER", "GOLD", "PLATINUM", "DIAMOND"]
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔮 Predict Churn", use_container_width=True):

    # Gender encoding
    gender_encoded = 1 if gender == "Male" else 0

    # Binary encoding
    has_cr_card_encoded = 1 if has_cr_card == "Yes" else 0
    active_member_encoded = 1 if active_member == "Yes" else 0

    # Geography encoding
    geography_germany = 1 if geography == "Germany" else 0
    geography_spain = 1 if geography == "Spain" else 0

    # Derived feature
    has_zero_balance = 1 if balance == 0 else 0

    # Card type encoding
    card_data = pd.DataFrame({
        "Card Type": [card_type]
    })

    card_encoded = ordinal_encoder.transform(card_data)[0][0]

    # Create input DataFrame
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [has_cr_card_encoded],
        "IsActiveMember": [active_member_encoded],
        "EstimatedSalary": [estimated_salary],
        "Satisfaction Score": [satisfaction_score],
        "Card Type": [card_encoded],
        "Point Earned": [point_earned],
        "Geography_Germany": [geography_germany],
        "Geography_Spain": [geography_spain],
        "HasZeroBalance": [has_zero_balance]
    })

    # Scale continuous features
    input_data[continuous_cols] = scaler.transform(
        input_data[continuous_cols]
    )

    # Ensure exact feature order
    input_data = input_data[feature_columns]

    # Churn probability
    churn_probability = model.predict_proba(input_data)[0][1]

    # Apply optimized threshold
    prediction = int(churn_probability >= threshold)

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{churn_probability:.1%}"
    )

    if prediction == 1:
        st.error("🔴 High Churn Risk")
        st.write(
            "This customer is predicted to be at risk of churn."
        )
    else:
        st.success("🟢 Low Churn Risk")
        st.write(
            "This customer is predicted to be unlikely to churn."
        )

    st.caption(
        f"Classification threshold used: {threshold:.2f}"
    )