import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(page_title="Heart Disease Prediction")

BASE_DIR = Path(__file__).resolve().parent

st.write("App started")

try:
    with open(BASE_DIR / "heart_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(BASE_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    st.success("Model and scaler loaded successfully")

except Exception as e:
    st.error(f"Loading failed: {e}")
    st.stop()

age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
chol = st.number_input("Cholesterol", 50, 700, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate", 50, 250, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, 0.1)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thal", [0, 1, 2])

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"Heart Disease Detected. Probability: {probability:.2%}")
    else:
        st.success(f"No Heart Disease Detected. Probability of disease: {probability:.2%}")