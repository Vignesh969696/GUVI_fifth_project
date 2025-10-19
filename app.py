import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from src.model import DepressionMLP

# Load preprocessing objects
scaler = joblib.load(r"D:\depression_prediction_project\saved_models\scaler.joblib")
label_encoders = joblib.load(r"D:\depression_prediction_project\saved_models\label_encoders.joblib")

# Load trained model
X_sample, _, _, _ = joblib.load(r"D:\depression_prediction_project\saved_models\data_split.joblib")
input_dim = X_sample.shape[1]
model = DepressionMLP(input_dim)
model.load_state_dict(torch.load(r"D:\depression_prediction_project\saved_models\model.pth"))
model.eval()

st.title("Depression Prediction App")
st.write("Enter your details to predict the likelihood of depression.")

# Create a form for user input
with st.form("input_form"):
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    City = st.text_input("City", "Enter city")
    Working_Professional_or_Student = st.selectbox("Working Professional or Student", ["Working Professional", "Student"])
    Profession = st.text_input("Profession", "Enter profession")
    Sleep_Duration = st.text_input("Sleep Duration", "Enter average hours")
    Dietary_Habits = st.text_input("Dietary Habits", "Enter dietary habits")
    Suicidal_Thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
    Family_History = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
    Work_Study_Hours = st.number_input("Work/Study Hours per day", min_value=0, max_value=24, value=8)
    Financial_Stress = st.number_input("Financial Stress (1-5)", min_value=1, max_value=5, value=3)
    
    submit = st.form_submit_button("Predict")

if submit:
    # Build input dataframe
    input_dict = {
        'Gender': [Gender],
        'Age': [Age],
        'City': [City],
        'Working Professional or Student': [Working_Professional_or_Student],
        'Profession': [Profession],
        'Sleep Duration': [Sleep_Duration],
        'Dietary Habits': [Dietary_Habits],
        'Have you ever had suicidal thoughts ?': [Suicidal_Thoughts],
        'Family History of Mental Illness': [Family_History],
        'Work/Study Hours': [Work_Study_Hours],
        'Financial Stress': [Financial_Stress]
    }
    
    input_df = pd.DataFrame(input_dict)
    
    # Encode categorical columns
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])
    
    # Scale numeric columns
    input_scaled = scaler.transform(input_df)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()
        prediction = 1 if prob >= 0.5 else 0
    
    st.write(f"Prediction: {'Depression' if prediction==1 else 'No Depression'}")
    st.write(f"Probability: {prob:.2f}")
