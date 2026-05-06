import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "heart_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

st.title("❤️ Heart Disease Risk Predictor")
st.markdown(
    "Enter patient clinical measurements to assess the likelihood of heart disease. "
    "This tool uses a machine learning model trained on the UCI Heart Disease Dataset."
)
st.divider()

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age        = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex_label  = st.selectbox("Sex", ["Female", "Male"])
    trestbps   = st.number_input("Resting Blood Pressure (mmHg)", min_value=50, max_value=250, value=120)
    chol       = st.number_input("Cholesterol Level (mg/dL)", min_value=50, max_value=700, value=200)
    fbs_label  = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", ["No", "Yes"])
    thalach    = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exang_label = st.selectbox("Exercise Induced Angina?", ["No", "Yes"])

with col2:
    cp_label = st.selectbox(
        "Chest Pain Type",
        ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"],
        help="Typical angina: chest pain related to reduced blood supply to the heart"
    )
    restecg_label = st.selectbox(
        "Resting ECG Result",
        ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"]
    )
    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0, max_value=10.0, value=1.0, step=0.1,
        help="ST depression induced by exercise relative to rest"
    )
    slope_label = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        ["Upsloping", "Flat", "Downsloping"]
    )
    ca_label = st.selectbox(
        "Major Vessels Colored by Fluoroscopy",
        [0, 1, 2, 3],
        help="Number of major vessels (0-3) colored by fluoroscopy"
    )
    thal_label = st.selectbox(
        "Thalassemia",
        ["Normal", "Fixed defect", "Reversible defect"],
        help="A blood disorder affecting haemoglobin production"
    )

sex       = 0 if sex_label == "Female" else 1
cp_map    = {"Typical angina": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3}
fbs       = 0 if fbs_label == "No" else 1
restecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
exang     = 0 if exang_label == "No" else 1
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map  = {"Normal": 0, "Fixed defect": 1, "Reversible defect": 2}

st.divider()

if st.button("Assess Heart Disease Risk", type="primary", use_container_width=True):

    input_df = pd.DataFrame([{
        "age"     : age,
        "sex"     : sex,
        "cp"      : cp_map[cp_label],
        "trestbps": trestbps,
        "chol"    : chol,
        "fbs"     : fbs,
        "restecg" : restecg_map[restecg_label],
        "thalach" : thalach,
        "exang"   : exang,
        "oldpeak" : oldpeak,
        "slope"   : slope_map[slope_label],
        "ca"      : ca_label,
        "thal"    : thal_map[thal_label]
    }])

    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    st.subheader("Risk Assessment Result")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Risk Probability", f"{probability:.1%}")
    with col_b:
        risk_level = "High Risk" if probability >= 0.5 else "Moderate Risk" if probability >= 0.3 else "Low Risk"
        st.metric("Risk Level", risk_level)
    with col_c:
        st.metric("Model Threshold", "50%")

    if prediction == 1:
        st.error(
            f"⚠️ **This patient is flagged as HIGH RISK for heart disease** "
            f"(probability: {probability:.1%}). Priority clinical follow-up is recommended."
        )
    else:
        st.success(
            f"✅ **This patient is currently LOW RISK for heart disease** "
            f"(probability: {probability:.1%}). Routine monitoring is advised."
        )

    st.divider()
    st.subheader("Key Risk Factors in This Assessment")

    factors = {
        "Age"              : (age,      20,  80,  "Risk increases significantly after 45"),
        "Cholesterol"      : (chol,     100, 400, "High cholesterol is a major risk factor"),
        "Max Heart Rate"   : (thalach,  70,  200, "Lower max heart rate may indicate reduced cardiac function"),
        "ST Depression"    : (oldpeak,  0,   6,   "Higher ST depression indicates more severe exercise-induced stress"),
        "Blood Pressure"   : (trestbps, 80,  200, "High resting BP increases cardiac workload"),
    }

    for factor, (value, low, high, note) in factors.items():
        normalized = min(max((value - low) / (high - low), 0.0), 1.0)
        st.write(f"**{factor}**: {value}")
        st.progress(normalized)
        st.caption(note)

    st.divider()

    with st.expander("View full input details"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption(
    "⚕️ This tool is for research and educational purposes only. "
    "Not a substitute for clinical diagnosis. "
    "All predictions should be reviewed by a qualified healthcare professional."
)
st.caption(
    "Built by Mubarak Adesola Adedeji · "
    "[LinkedIn](https://linkedin.com/in/mubarak-adedeji-776804273) · "
    "[GitHub](https://github.com/Mubydeji)"
)
