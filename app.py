import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "heart_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

st.title("Heart Disease Risk Predictor")
st.write("Fill in the patient's details below to estimate the likelihood of heart disease.")

st.subheader("Basic Information")
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex_label = st.selectbox("Sex", ["Female", "Male"])

st.subheader("Symptoms and Medical Condition")
cp_label = st.selectbox(
    "Chest Pain Type",
    [
        "Typical angina",
        "Atypical angina",
        "Non-anginal pain",
        "Asymptomatic"
    ],
    help="Choose the type of chest pain the patient experiences."
)

trestbps = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=50,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol Level (mg/dl)",
    min_value=50,
    max_value=700,
    value=200
)

fbs_label = st.selectbox(
    "Fasting Blood Sugar Above 120 mg/dl?",
    ["No", "Yes"]
)

restecg_label = st.selectbox(
    "Resting ECG Result",
    [
        "Normal",
        "ST-T wave abnormality",
        "Left ventricular hypertrophy"
    ]
)

thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=50,
    max_value=250,
    value=150
)

exang_label = st.selectbox(
    "Exercise Induced Angina?",
    ["No", "Yes"]
)

oldpeak = st.number_input(
    "ST Depression (Oldpeak)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1,
    help="A measure of ST depression induced by exercise relative to rest."
)

slope_label = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    ["Upsloping", "Flat", "Downsloping"]
)

ca_label = st.selectbox(
    "Number of Major Vessels Colored by Fluoroscopy",
    [0, 1, 2, 3]
)

thal_label = st.selectbox(
    "Thalassemia Test Result",
    ["Normal", "Fixed defect", "Reversible defect"]
)

sex = 0 if sex_label == "Female" else 1
cp_map = {
    "Typical angina": 0,
    "Atypical angina": 1,
    "Non-anginal pain": 2,
    "Asymptomatic": 3
}
fbs = 0 if fbs_label == "No" else 1
restecg_map = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2
}
exang = 0 if exang_label == "No" else 1
slope_map = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}
thal_map = {
    "Normal": 0,
    "Fixed defect": 1,
    "Reversible defect": 2
}

if st.button("Predict Risk"):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp_map[cp_label],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg_map[restecg_label],
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope_map[slope_label],
        "ca": ca_label,
        "thal": thal_map[thal_label]
    }])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"High likelihood of heart disease")
    else:
        st.success(f"Low likelihood of heart disease")

    st.write(f"Estimated probability of heart disease: **{probability:.2%}**")

    with st.expander("See entered details"):
        st.dataframe(input_df, use_container_width=True)
