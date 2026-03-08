import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_artifacts():
    try:
        with open(BASE_DIR / "heart_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(BASE_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Artifact loading failed: {e}")
        st.stop()

model, scaler = load_artifacts()

st.title("Heart Disease Prediction App")
st.write("Enter the patient's clinical details below to predict heart disease.")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol", min_value=50, max_value=700, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "False" if x == 0 else "True")
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3])
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

    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"Prediction: Heart Disease Detected\nProbability: {probability:.2%}")
        else:
            st.success(f"Prediction: No Heart Disease Detected\nProbability of Disease: {probability:.2%}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
