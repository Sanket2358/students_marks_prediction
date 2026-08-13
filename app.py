import streamlit as st
import pandas as pd
import pickle
import time

# ==========================================
# 1. Page Configuration & Setup
# ==========================================
st.set_page_config(
    page_title="EduPredict Pro | AI Student Analytics", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. Advanced CSS with Dark/Light Mode Fix & Animations
# ==========================================
# Using CSS variables like var(--text-color) ensures it works flawlessly in both Dark and Light modes.
st.markdown("""
    <style>
    /* Fade-in Animation for the whole app */
    .block-container {
        animation: fadeIn 1.2s ease-in-out;
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* Custom Gradient Button (Works well in both modes) */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    div.stButton > button:first-child:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 15px rgba(124, 58, 237, 0.4);
    }

    /* Metric Cards Styling */
    div[data-testid="metric-container"] {
        background-color: rgba(124, 58, 237, 0.1);
        border: 1px solid rgba(124, 58, 237, 0.2);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #7c3aed;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. Model Loading
# ==========================================
@st.cache_resource(show_spinner=False)
def load_model():
    with open("naive_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ `naive_model.pkl` file not found. Ensure it is in the same directory.")
    st.stop()

# ==========================================
# 4. Sidebar: Software Navigation & Dev Info
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/10817/10817292.png", width=80)
    st.title("EduPredict Pro")
    st.caption("v2.0 | AI Analytics Engine")
    st.markdown("---")
    
    st.markdown("### 🛠️ About Software")
    st.info(
        "This software leverages a Gaussian Naive Bayes classification algorithm to evaluate student metrics "
        "and predict academic outcomes with high accuracy."
    )
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer Profile")
    st.markdown("**Role:** College Student & AI/ML Developer")
    st.markdown(
        "Built applying advanced machine learning concepts and skills gained from experiences like the **Google AIML Virtual Internship**."
    )
    st.caption("Balancing 9-to-5 college lectures with building cool AI projects! 🚀")

# ==========================================
# 5. Main UI: Software Header & Tabs
# ==========================================
st.title("🧠 Student Success Prediction Engine")
st.markdown("Enter student data to run the AI simulation and forecast academic results.")

# Creating Tabs for a multi-page software feel
tab1, tab2 = st.tabs(["📊 Prediction Dashboard", "📈 Model Insights"])

with tab1:
    st.markdown("### 📋 Student Metrics Input")
    
    # Using expanders to organize UI nicely
    with st.expander("👤 Personal & Demographic Details", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=10, max_value=100, value=18)
        with c2:
            gender_input = st.selectbox("Gender", options=["Male", "Female"])
            gender = 0 if gender_input == "Male" else 1
        with c3:
            department = st.number_input("Department ID", min_value=0, max_value=20, value=1, help="Integer ID representing the student's department.")

    with st.expander("📚 Academic Performance & Habits", expanded=True):
        c4, c5 = st.columns(2)
        with c4:
            study_hours = st.slider("Daily Study Hours", 0.0, 24.0, 4.0, 0.5, help="Average hours spent studying outside of college.")
            attendance = st.slider("Attendance (%)", 0.0, 100.0, 85.0, 1.0)
        with c5:
            assignments = st.number_input("Assignments Completed", 0, 100, 15)
            midterm = st.number_input("Midterm Score (/100)", 0.0, 100.0, 75.0)
            final = st.number_input("Final Target/Expected Score (/100)", 0.0, 100.0, 80.0)

    st.markdown("<br>", unsafe_allow_html=True) # Spacer

    # ==========================================
    # 6. Prediction Engine & Animations
    # ==========================================
    if st.button("⚙️ Run AI Analysis"):
        
        # Software-like progress bar animation
        progress_text = "Processing data through Naive Bayes algorithm..."
        my_bar = st.progress(0, text=progress_text)
        
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        
        time.sleep(0.3)
        my_bar.empty() # Remove progress bar after completion
        
        # Prepare Data
        data = {
            "Age": age, "Gender": gender, "Department": department,
            "Study_Hours_Per_Day": study_hours, "Attendance_Percentage": attendance,
            "Assignments_Completed": assignments, "Midterm_Score": midterm, "Final_Score": final
        }
        input_df = pd.DataFrame([data])
        
        # Prediction
        try:
            prediction = model.predict(input_df)[0]
            
            st.markdown("---")
            st.markdown("### 🎯 Analysis Result")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                if prediction == "Pass":
                    st.success("✅ OUTCOME: PASS")
                    st.balloons()
                else:
                    st.error("❌ OUTCOME: FAIL")
                    
            with res_col2:
                if prediction == "Pass":
                    st.info("💡 **AI Recommendation:** The student shows a strong trajectory. Maintaining current attendance and study hours is advised.")
                else:
                    st.warning("⚠️ **AI Alert:** The student is at risk. Immediate intervention required in attendance and daily study hours to improve the final outcome.")
            
            # Show Metrics Dashboard
            st.markdown("#### Key Drivers")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(label="Attendance", value=f"{attendance}%", delta="Critical" if attendance < 75 else "Optimal")
            m2.metric(label="Study Hours", value=f"{study_hours} hrs")
            m3.metric(label="Assignments", value=assignments)
            m4.metric(label="Current Avg", value=f"{(midterm + final) / 2}%")

        except Exception as e:
            st.error(f"Software Error during prediction matrix computation: {e}")

# ==========================================
# 7. Model Insights Tab
# ==========================================
with tab2:
    st.markdown("### 🔬 Inside the AI Engine")
    st.write("This software utilizes **Gaussian Naive Bayes (GaussianNB)** from the `scikit-learn` library.")
    st.write("The model assumes that the continuous values associated with each class are distributed according to a normal (or Gaussian) distribution.")
    
    st.info("""
    **Model Input Features:**
    1. Age (Continuous)
    2. Gender (Categorical -> Encoded)
    3. Department (Categorical -> Encoded)
    4. Study Hours Per Day (Continuous)
    5. Attendance Percentage (Continuous)
    6. Assignments Completed (Discrete)
    7. Midterm Score (Continuous)
    8. Final Score (Continuous)
    """)
    st.caption("Model loaded successfully from `naive_model.pkl` (Scikit-Learn v1.6.1)")
