import streamlit as st
import pandas as pd
import joblib
import numpy as np


st.set_page_config(page_title="Academic Performance Predictor", layout="centered")
st.title(" Gaming & Academic Performance Predictor")


def load_assets():
    xgb_gs = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    ct = joblib.load('encoder.pkl')
    return xgb_gs, scaler, ct

xgb_gs, scaler, ct = load_assets()

st.subheader(" Student Habit Inputs")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Student Age")
    study_hours = st.number_input("Weekly Study Hours")
    gaming_hours = st.number_input("Weekly Gaming Hours")
    device_usage = st.number_input("Daily Device Screen Time (Hours)")
    attendance = st.number_input("School Attendance Percentage (%)")
    

with col2:
    sleep_hours = st.number_input("Average Sleep Hours per Night")
    social_activity = st.number_input("Weekly Social Activity (Hours)")
    reaction_time_ms = st.number_input("Reaction Time (ms)")
    addiction_score = st.number_input("Gaming Addiction Assessment Score")
    
    
    stress_input = st.selectbox("Current Stress Level", options=["Low", "Medium", "High"])
    stress_mapping = {"Low": 0, "Medium": 1, "High": 2}
    stress_level = stress_mapping[stress_input]

gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
gaming_genre = st.selectbox("Favorite Gaming Genre", options=["FPS", "RPG", "Casual"])


if st.button(" Predict Final Student Grade", type="primary"):
    
    
    raw_data = {
        'age': [age],
        'study_hours': [study_hours],
        'gaming_hours': [gaming_hours],
        'device_usage': [device_usage],
        'attendance': [attendance],
        'sleep_hours': [sleep_hours],
        'social_activity': [social_activity],
        'reaction_time_ms': [reaction_time_ms],
        'addiction_score': [addiction_score],
        'stress_level': [stress_level],
        'gender': [gender],
        'gaming_genre': [gaming_genre]
    }
    input_df = pd.DataFrame(raw_data)
    
   
    encoded_data = ct.transform(input_df)
    scaled_data = scaler.transform(encoded_data)

       
    prediction = xgb_gs.predict(scaled_data)
    
   
    raw_score = float(prediction[0])
    
    
    final_grade = max(0.0, min(100.0, raw_score))
    
   
    st.success(f"### Predicted Academic Grade Score: **{final_grade:.2f}**")

    
   

