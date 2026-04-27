import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(page_title="Loan Risk AI", page_icon="🏦", layout="centered")

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2c3e50; color: white; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_models():
    # Replace these with your actual filenames
    model = joblib.load('loan_default_model.pkl')
    scaler = joblib.load('loan_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_models()
except:
    st.error("Model files not found! Please ensure 'loan_default_model.pkl' and 'loan_scaler.pkl' are in the folder.")

# --- APP HEADER ---
st.title("🏦 Loan Default Prediction System")
st.markdown("### Decision Support Tool for Financial Risk Assessment")
st.write("Fill in the applicant's details below to generate a risk profile.")

# --- INPUT UI ---
with st.container():
    st.subheader("Personal & Financial Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Annual Income ($)", min_value=0, value=50000)
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=15000)
        credit_score = st.slider("Credit Score", 300, 850, 650)

    with col2:
        months_employed = st.number_input("Months at Current Job", min_value=0, value=24)
        interest_rate = st.number_input("Interest Rate (%)", 0.0, 35.0, 10.5)
        loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60])
        num_credit_lines = st.number_input("Number of Credit Lines", 1, 20, 3)

st.divider()

with st.container():
    st.subheader("Employment & Background")
    col3, col4 = st.columns(2)

    with col3:
        education = st.selectbox("Highest Education", ["High School", "Bachelor's", "Master's", "PhD"])
        employment_type = st.selectbox("Employment Sector", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        
    with col4:
        loan_purpose = st.selectbox("Purpose of Loan", ["Auto", "Business", "Education", "Home", "Other"])
        has_mortgage = st.radio("Has Existing Mortgage?", ["No", "Yes"])
        has_dependents = st.radio("Has Dependents?", ["No", "Yes"])
        has_cosigner = st.radio("Has a Co-Signer?", ["No", "Yes"])

# --- DATA PREPROCESSING FOR PREDICTION ---
# Mapping inputs to the numerical values used in training
edu_map = {"Bachelor's": 0, "High School": 1, "Master's": 2, "PhD": 3}
emp_map = {"Full-time": 0, "Part-time": 1, "Self-employed": 2, "Unemployed": 3}
mar_map = {"Divorced": 0, "Married": 1, "Single": 2}
pur_map = {"Auto": 0, "Business": 1, "Education": 2, "Home": 3, "Other": 4}
binary_map = {"No": 0, "Yes": 1}

# Calculate DTI Ratio automatically
dti_ratio = (loan_amount / loan_term) / (income / 12) if income > 0 else 0

# --- PREDICTION LOGIC ---
if st.button("🚀 ANALYZE RISK PROFILE"):
    
    # Constructing feature array in the exact order as training
    # Order: Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio, 
    # Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner
    
    raw_data = np.array([[
        age, income, loan_amount, credit_score, months_employed, 
        num_credit_lines, interest_rate, loan_term, dti_ratio,
        edu_map[education], emp_map[employment_type], mar_map[marital_status],
        binary_map[has_mortgage], binary_map[has_dependents], 
        pur_map[loan_purpose], binary_map[has_cosigner]
    ]])

    # Scaling
    scaled_data = scaler.transform(raw_data)
    
    # Prediction
    prediction = model.predict(scaled_data)
    prob = model.predict_proba(scaled_data)[0][1]

    # --- DISPLAY RESULTS ---
    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"### Result: ⚠️ HIGH RISK OF DEFAULT")
        st.write(f"The model has calculated a **{prob:.1%}** probability of loan default.")
        st.progress(prob)
    else:
        st.success(f"### Result: ✅ LOW RISK / SAFE")
        st.write(f"The model predicts a **{1-prob:.1%}** probability of successful repayment.")
        st.progress(prob)

st.markdown("<br><hr><center>Developed by Aastha Shinde | BIA Capstone Project</center>", unsafe_allow_html=True)
