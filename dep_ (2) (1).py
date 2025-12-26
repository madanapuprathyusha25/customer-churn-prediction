import joblib
import numpy as np
import streamlit as st

# 1. LOAD THE SAVED MODEL AND SCALER 
# Use relative paths. Ensure these files are in the same folder as your app.py.
try:
    # --- THIS IS THE CHANGE ---
    model = joblib.load("best_model.pkl") 
    # -------------------------
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}. Make sure 'best_model.pkl' and 'scaler.pkl' are in the correct directory.")
    st.stop()

# Load the trained model
#model = joblib.load(r"E:\New DS projects\Excelr\Tele Communications\team work\model\rf_model.pkl")

# Custom CSS Background
url = "https://i.pinimg.com/originals/5c/39/55/5c3955b34e9bc66aa693eadd8f59db54.jpg"
custom_css = f"""
    <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("{url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
            color: white;
        }}
        .stButton > button {{
            border-radius: 12px;
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 8px 16px;
        }}
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Title
st.title("Customer Churn Prediction App")
st.write("Fill in the details below to predict whether a customer is likely to churn.")

# Define Input Features (13 features, same as training)
account_length = st.number_input("Account Length (days):", min_value=0, step=1)
voice_mail_plan = st.selectbox("Voice Mail Plan:", ["No", "Yes"])
voice_mail_messages = st.number_input("Number of Voice Mail Messages:", min_value=0, step=1)
international_mins = st.number_input("International Minutes:", min_value=0.0, step=0.1)
customer_service_calls = st.number_input("Customer Service Calls:", min_value=0, step=1)
international_plan = st.selectbox("International Plan:", ["No", "Yes"])
day_charge = st.number_input("Day Charge:", min_value=0.0, step=0.01)
evening_charge = st.number_input("Evening Charge:", min_value=0.0, step=0.01)
night_charge = st.number_input("Night Charge:", min_value=0.0, step=0.01)
international_calls = st.number_input("International Calls:", min_value=0, step=1)
international_charge = st.number_input("International Charge:", min_value=0.0, step=0.01)
total_charge = st.number_input("Total Charge:", min_value=0.0, step=0.01)
total_mins = st.number_input("Total Minutes:", min_value=0.0, step=0.1)

# Encode categorical features
voice_mail_plan_val = 1 if voice_mail_plan == "Yes" else 0
international_plan_val = 1 if international_plan == "Yes" else 0

# Prediction
if st.button("Predict Churn"):
    input_data = np.array([[
        account_length,
        voice_mail_plan_val,
        voice_mail_messages,
        international_mins,
        customer_service_calls,
        international_plan_val,
        day_charge,
        evening_charge,
        night_charge,
        international_calls,
        international_charge,
        total_charge,
        total_mins
    ]])

    # Predict
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    # Map prediction
    mapping = {0: "Not Churned", 1: "Churned"}
    result = mapping.get(prediction[0], "Unknown")

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Prediction:** {result}")
    st.write(f"**Churn Probability:** {prob:.2f}")

    if result == "Churned":
        st.error("⚠️ Customer is likely to churn!")
    else:
        st.success("✅ Customer is not likely to churn.")
        st.balloons()
