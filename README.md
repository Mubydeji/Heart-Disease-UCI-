# Heart Disease Risk Predictor
### Clinical Risk Assessment Powered by Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project builds a machine learning classification pipeline to predict heart disease risk from patient clinical measurements. It is deployed as an interactive web application where clinicians or researchers can enter patient data and receive an instant risk assessment with probability scoring and factor-level visualisation.

---

## Live Demo

🔗 [dsfp5app.streamlit.app](https://dsfp5app.streamlit.app)

---

## Clinical Question

> *Given a patient's clinical measurements, can we accurately predict whether they are at risk of heart disease?*

---

## Dataset

**UCI Heart Disease Dataset**
- 303 patient records
- 13 clinical features including age, chest pain type, cholesterol, ECG results, and thalassemia
- Target variable: presence of heart disease (1 = present, 0 = absent)
- Source: UCI Machine Learning Repository

---

## Project Structure

    heart-disease-risk-predictor/
    ├── app.py                          # Streamlit application
    ├── heart-disease-prediction.ipynb  # Full training notebook
    ├── heart_model.pkl                 # Trained model
    ├── scaler.pkl                      # Fitted StandardScaler
    ├── requirements.txt
    └── README.md

---

## Methodology

### 1. Data Cleaning and Preprocessing
- Handled missing values and outliers across 13 clinical features
- Encoded categorical variables (chest pain type, ECG result, thalassemia)
- Applied StandardScaler for feature normalisation

### 2. Feature Engineering
- Engineered interaction terms between high-signal features
- Derived composite risk indicators from correlated clinical measurements

### 3. Model Training
- Trained and evaluated multiple classification models
- Selected best model based on ROC-AUC and clinical recall requirements
- Serialised final model and scaler as pickle files for deployment

### 4. Deployment
- Built interactive Streamlit UI with two-column layout
- Risk probability displayed with three-tier classification: Low, Moderate, High
- Per-factor progress bars with clinical context for each key measurement
- Expandable input detail view for full record inspection

---

## App Features

- Enter 13 clinical measurements via an intuitive form interface
- Instant risk probability and risk level classification
- Five key risk factor visualisations with clinical context
- Clear disclaimer distinguishing research use from clinical diagnosis

---

## Key Risk Factors

Based on model coefficients, the strongest predictors of heart disease in this dataset are:

- Chest pain type — asymptomatic presentation carries highest risk
- Number of major vessels colored by fluoroscopy
- Thalassemia type — reversible defect is a strong positive predictor
- ST depression (Oldpeak) — higher values indicate greater exercise stress
- Maximum heart rate achieved — lower values associated with increased risk

---

## Tools and Libraries

| Tool | Purpose |
|---|---|
| Python | Core language |
| Scikit-learn | Model training and evaluation |
| Pandas / NumPy | Data manipulation |
| Streamlit | UI and deployment |
| Pickle | Model serialisation |

---

## Author

**Mubarak Adesola Adedeji**
Data Analyst · AI Developer | Python · SQL · R · Power BI
[LinkedIn](https://linkedin.com/in/mubarak-adedeji-776804273) · [GitHub](https://github.com/Mubydeji)

---

## License

MIT License — free to use, adapt, and build on with attribution.
