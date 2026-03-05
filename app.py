import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def load_model():
    with open('heart_disease_model.pkl', 'rb') as f:
        return pickle.load(f)

bundle  = load_model()
model   = bundle['model']
scaler  = bundle['scaler']
metrics = bundle['metrics']

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease Prediction System")
st.markdown("Enter patient medical details below to predict heart disease risk.")
st.divider()

with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("Accuracy",    f"{metrics['accuracy']*100:.1f}%")
    st.metric("Recall",      f"{metrics['recall']*100:.1f}%")
    st.metric("Precision",   f"{metrics['precision']*100:.1f}%")
    st.metric("F1 Score",    f"{metrics['f1']*100:.1f}%")
    st.metric("ROC-AUC",     f"{metrics['roc_auc']:.3f}")
    st.metric("CV Accuracy", f"{metrics['cv_mean']*100:.1f}% ± {metrics['cv_std']*100:.1f}%")
    st.info("Logistic Regression | L2 Regularization | StandardScaler")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Patient Info")
    Age         = st.slider("Age", 20, 90, 55)
    Sex         = st.selectbox("Sex", ["M", "F"])
    RestingBP   = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 130)
    Cholesterol = st.slider("Cholesterol (mg/dl)", 0, 600, 245)

with col2:
    st.subheader("🩺 Clinical Data")
    ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    MaxHR         = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    Oldpeak       = st.slider("ST Depression (Oldpeak)", -3.0, 6.0, 1.0, step=0.1)
    ExerciseAngina= st.selectbox("Exercise Induced Angina", ["N", "Y"])

with col3:
    st.subheader("🔬 Test Results")
    FastingBS   = st.selectbox("Fasting Blood Sugar > 120", ["0 - No", "1 - Yes"])
    RestingECG  = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    ST_Slope    = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.divider()

if st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True):

    input_data = pd.DataFrame([{
        'Age':            Age,
        'Sex':            1 if Sex == "M" else 0,
        'ChestPainType':  ["ATA","NAP","ASY","TA"].index(ChestPainType),
        'RestingBP':      RestingBP,
        'Cholesterol':    Cholesterol,
        'FastingBS':      int(FastingBS[0]),
        'RestingECG':     ["Normal","ST","LVH"].index(RestingECG),
        'MaxHR':          MaxHR,
        'ExerciseAngina': 1 if ExerciseAngina == "Y" else 0,
        'Oldpeak':        Oldpeak,
        'ST_Slope':       ["Up","Flat","Down"].index(ST_Slope)
    }])

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    r1, r2 = st.columns(2)
    with r1:
        if prediction == 1:
            st.error("## ⚠️ Heart Disease DETECTED")
            st.error(f"### Risk: {probability*100:.1f}%")
            st.warning("Please consult a cardiologist immediately.")
        else:
            st.success("## ✅ No Heart Disease Detected")
            st.success(f"### Risk: {probability*100:.1f}%")
            st.info("Routine checkups recommended.")
    with r2:
        st.subheader("📋 Patient Summary")
        st.write(f"**Age:** {Age} | **Sex:** {'Male' if Sex=='M' else 'Female'}")
        st.write(f"**Cholesterol:** {Cholesterol} | **BP:** {RestingBP}")
        st.write(f"**Max HR:** {MaxHR} | **ST Depression:** {Oldpeak}")
        st.progress(float(probability), text=f"Disease Probability: {probability*100:.1f}%")

    st.caption("⚠️ For educational use only. Always consult a doctor.")