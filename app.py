import streamlit as st
import pandas as pd
import pickle
import time

# 1. Page Configuration (Wide layout for better responsiveness)
st.set_page_config(
    page_title="Student Success Predictor", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Advanced UI & Color Combinations
st.markdown("""
    <style>
    /* Main background and text color */
    .stApp {
        background-color: #f4f6f9;
        color: #1e1e1e;
    }
    
    /* Styled Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Custom Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Input field styling */
    .stNumberInput, .stSelectbox {
        background-color: white;
        border-radius: 5px;
        padding: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load Model with Caching
@st.cache_resource
def load_model():
    with open("naive_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ `naive_model.pkl` file not found. Please ensure it is in the same directory.")
    st.stop()

# 4. Sidebar for App Info & Settings
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135810.png", width=100) # Placeholder icon
    st.title("About the App")
    st.info("This application uses an AI/ML Naive Bayes model to predict student performance based on historical academic and behavioral data.")
    st.markdown("---")
    st.write("### Model Features:")
    st.write("✔️ Age & Gender")
    st.write("✔️ Study Hours & Attendance")
    st.write("✔️ Midterm & Final Scores")

# 5. Main Content Area
st.title("🎓 Student Performance Predictor")
st.markdown("Enter the student's metrics below to generate an AI-driven prediction on their final outcome (**Pass** or **Fail**).")
st.markdown("---")

# 6. Grouped Inputs for Better UX (Using Containers and Columns)
with st.container():
    st.subheader("👤 Demographics & Basic Info")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=18)
    with col2:
        gender_input = st.selectbox("Gender", options=["Male", "Female"])
        gender = 0 if gender_input == "Male" else 1
    with col3:
        department = st.number_input("Department ID", min_value=0, max_value=20, value=0, help="Enter the integer code for the department.")

with st.container():
    st.subheader("📚 Academic Metrics")
    col4, col5 = st.columns(2)
    
    with col4:
        study_hours = st.slider("Study Hours Per Day", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
        attendance = st.slider("Attendance Percentage (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
    with col5:
        assignments = st.number_input("Assignments Completed", min_value=0, max_value=100, value=10)
        midterm = st.number_input("Midterm Score", min_value=0.0, max_value=100.0, value=75.0)
        final = st.number_input("Final Score", min_value=0.0, max_value=100.0, value=80.0)

st.markdown("---")

# 7. Prediction Logic with Interactive UI
if st.button("🚀 Predict Outcome"):
    
    # Show a loading spinner to make it feel advanced
    with st.spinner("Analyzing student data using AI/ML..."):
        time.sleep(1) # Fake delay for visual effect
        
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
        
        input_df = pd.DataFrame([data])
        
        try:
            prediction = model.predict(input_df)[0]
            
            # Display results in an attractive metric box
            st.markdown("### Prediction Result")
            if prediction == "Pass":
                st.success("🎉 **OUTCOME: PASS**")
                st.balloons() # Fun Streamlit animation
                st.info("The model indicates strong academic performance. Keep up the good work!")
            else:
                st.error("⚠️ **OUTCOME: FAIL**")
                st.warning("The model predicts a risk of failing. Consider increasing study hours and attendance.")
                
            # Show a quick summary of key driving factors
            st.markdown("#### Input Summary:")
            scol1, scol2, scol3 = st.columns(3)
            scol1.metric("Attendance", f"{attendance}%")
            scol2.metric("Study Hours", f"{study_hours} hrs")
            scol3.metric("Total Score", f"{(midterm + final) / 2}% (Avg)")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
