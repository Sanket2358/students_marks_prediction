import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# Load the trained Naive Bayes model
@st.cache_resource
def load_model():
    with open("naive_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Header
st.title("🎓 Student Performance Predictor")
st.write("Enter the student's metrics below to predict if they will **Pass** or **Fail**.")

st.markdown("---")

# Input fields arranged in columns for a cleaner UI
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=18)
    
    # Assuming Gender was label-encoded (e.g., 0 for Male, 1 for Female)
    # Update the display options below based on your actual training preprocessing!
    gender_input = st.selectbox("Gender", options=["Male (0)", "Female (1)"])
    gender = 0 if "0" in gender_input else 1
    
    # Assuming Department was label-encoded as integers (0, 1, 2, etc.)
    department = st.number_input("Department (Encoded ID)", min_value=0, max_value=20, value=0, help="Enter the integer code for the department used during training.")
    
    study_hours = st.number_input("Study Hours Per Day", min_value=0.0, max_value=24.0, value=3.0)

with col2:
    attendance = st.number_input("Attendance Percentage (%)", min_value=0.0, max_value=100.0, value=85.0)
    assignments = st.number_input("Assignments Completed", min_value=0, max_value=100, value=10)
    midterm = st.number_input("Midterm Score", min_value=0.0, max_value=100.0, value=75.0)
    final = st.number_input("Final Score", min_value=0.0, max_value=100.0, value=80.0)

st.markdown("---")

# Prediction Button
if st.button("Predict Outcome", type="primary"):
    # 1. Gather inputs into a dictionary matching your model's exact feature names
    data = {
        "Age": age,
        "Gender": gender,
        "Department": department,
        "Study_Hours_Per_Day": study_hours,
        "Attendance_Percentage": attendance,
        "Assignments_Completed": assignments,
        "Midterm_Score": midterm,
        "Final_Score": final
    }
    
    # 2. Convert to DataFrame (models prefer DataFrames over 2D arrays to match feature names)
    input_df = pd.DataFrame([data])
    
    # 3. Predict
    try:
        prediction = model.predict(input_df)[0]
        
        # 4. Display result
        if prediction == "Pass":
            st.success("🎉 Prediction: **PASS**")
        else:
            st.error("⚠️ Prediction: **FAIL**")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
