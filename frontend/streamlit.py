import streamlit as st
import requests
import pandas as pd
import os

st.set_page_config(page_title="Final Project Model Deployment - Obesity Level Prediction")

API_BASE_URL = "https://uas-model-deployment-backend-production-0313.up.railway.app"

def check_api():
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_obesity(data):
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=data, timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json()
    except Exception as e:
        return False, {"detail": str(e)}

st.title("Final Project Model Deployment - Obesity Level Prediction")
st.write("""This application predicts obesity level.
\n\tMade by Octavius Sandriago - 2702221135""")

if not check_api():
    st.error("⚠️ API not running. Please start the FastAPI server first.")
    st.code("uvicorn fastAPI:app --reload")
    st.stop()

st.success("✅ Connected to prediction API")

st.header("Enter Your Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    height = st.number_input("Height (meters)", min_value=1.0, max_value=2.5, value=1.7, step=0.01)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    gender = st.selectbox("Gender", ["male", "female"])

with col2:
    family_history = st.selectbox("Family history of overweight?", ["no", "yes"])
    smoke = st.selectbox("Do you smoke?", ["no", "yes"])
    favc = st.selectbox("Eat high caloric food frequently?", ["no", "yes"])
    scc = st.selectbox("Monitor calories?", ["no", "yes"])

st.subheader("Daily Habits")

col1, col2 = st.columns(2)

with col1:
    fcvc = st.slider("Vegetable consumption frequency", 1.0, 3.0, 2.0, 0.1)
    ncp = st.slider("Number of main meals", 1.0, 4.0, 3.0, 0.1)
    ch2o = st.slider("Water consumption (liters/day)", 1.0, 3.0, 2.0, 0.1)
    faf = st.slider("Physical activity frequency", 0.0, 3.0, 1.0, 0.1)

with col2:
    tue = st.slider("Technology use (hours/day)", 0.0, 2.0, 1.0, 0.1)
    caec = st.selectbox("Eat between meals?", ["no", "sometimes", "frequently", "always"])
    calc = st.selectbox("Alcohol consumption", ["no", "sometimes", "frequently", "always"])
    mtrans = st.selectbox("Main transportation", ["walking", "bike", "public_transportation", "automobile", "motorbike"])

if st.button("Predict Obesity Level", type="primary"):
    input_data = {
        "Age": float(age),
        "Height": float(height),
        "Weight": float(weight),
        "FCVC": float(fcvc),
        "NCP": float(ncp),
        "CH2O": float(ch2o),
        "FAF": float(faf),
        "TUE": float(tue),
        "Gender": gender,
        "family_history_with_overweight": family_history,
        "FAVC": favc,
        "SMOKE": smoke,
        "SCC": scc,
        "MTRANS": mtrans,
        "CAEC": caec,
        "CALC": calc
    }
    
    with st.spinner("Predicting..."):
        success, result = predict_obesity(input_data)
    
    if success:
        st.header("Results")
        
        prediction = result['predicted_obesity_level'].replace('_', ' ').title()
        confidence = result['confidence']
        
        st.success(f"**Predicted Obesity Level: {prediction}**")
        st.info(f"Confidence: {confidence}%")
        
        st.subheader("All Probabilities")
        probs = result['probabilities']
        for level, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            level_name = level.replace('_', ' ').title()
            st.write(f"• {level_name}: {prob}%")
            
    else:
        st.error("Prediction failed!")
        st.error(result.get('detail', 'Unknown error'))