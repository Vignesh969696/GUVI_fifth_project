import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
from model import DepressionMLP

# -------------------------------
# Load preprocessing artifacts
# -------------------------------
scaler = joblib.load("scaler.joblib")
ohe = joblib.load("onehot_encoder.joblib")
model_path = "model.pth"

# -------------------------------
# Device setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Compute input_dim for the model
# -------------------------------
num_features = 5  # Age, Work Pressure, Job Satisfaction, Work/Study Hours, Financial Stress
# Using a dummy row to get number of categorical features from one-hot encoder
cat_features = ohe.transform([["Unknown"]*6]).shape[1]  # 6 categorical columns
input_dim = num_features + cat_features

# -------------------------------
# Load trained model
# -------------------------------
model = DepressionMLP(input_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Depression Prediction App")
st.write("Enter your details to predict the likelihood of depression.")

with st.form("input_form"):
    Gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    Working_Professional_or_Student = st.selectbox("Working Professional or Student", ["Working Professional", "Student"])
    Profession = st.text_input("Profession", "Enter profession")
    Sleep_Duration = st.selectbox("Sleep Duration", ["<5 hours", "5-6 hours", "6-7 hours", "7-8 hours", ">8 hours"])
    Dietary_Habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
    Suicidal_Thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
    Family_History = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
    Work_Study_Hours = st.number_input("Work/Study Hours per day", min_value=0, max_value=24, value=8)
    Financial_Stress = st.number_input("Financial Stress (1-5)", min_value=1, max_value=5, value=3)
    submit = st.form_submit_button("Predict")

if submit:
    # -------------------------------
    # Build input dataframe
    # -------------------------------
    input_dict = {
        'Gender': [1 if Gender == "Male" else (0 if Gender == "Female" else 2)],
        'Age': [Age],
        'Working Professional or Student': [Working_Professional_or_Student],
        'Profession': [Profession],
        'Sleep Duration': [Sleep_Duration],
        'Dietary Habits': [Dietary_Habits],
        'Have you ever had suicidal thoughts ?': [1 if Suicidal_Thoughts == "Yes" else 0],
        'Family History of Mental Illness': [1 if Family_History == "Yes" else 0],
        'Work/Study Hours': [Work_Study_Hours],
        'Financial Stress': [Financial_Stress],
        # Fill missing columns expected by one-hot encoder
        'City': ["Unknown"],
        'Degree': ["Unknown"]
    }

    df_input = pd.DataFrame(input_dict)

    # -------------------------------
    # One-hot encode categorical features
    # -------------------------------
    multi_cat_cols = ['City', 'Working Professional or Student', 'Profession', 'Sleep Duration', 'Dietary Habits', 'Degree']
    X_cat = ohe.transform(df_input[multi_cat_cols])

    # -------------------------------
    # Scale numeric features (5 numeric columns)
    # -------------------------------
    numeric_data = np.array([[Age, 0, 0, Work_Study_Hours, Financial_Stress]])  # 0 for Work Pressure & Job Satisfaction
    numeric_scaled = scaler.transform(numeric_data)

    # -------------------------------
    # Combine numeric + categorical
    # -------------------------------
    X_final = np.hstack([numeric_scaled, X_cat])
    X_tensor = torch.tensor(X_final, dtype=torch.float32).to(device)

    # -------------------------------
    # Predict
    # -------------------------------
    with torch.no_grad():
        logits = model(X_tensor)
        prob = torch.sigmoid(logits).item()
        prediction = 1 if prob >= 0.5 else 0

    # -------------------------------
    # Show results
    # -------------------------------
    st.subheader("Prediction Result")
    st.write(f"**Prediction:** {'Depression' if prediction == 1 else 'No Depression'}")
    st.write(f"**Probability:** {prob:.2f}")

