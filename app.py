import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------
# Load trained models (PIPELINES)
# -------------------------------------------------
# IMPORTANT:
# Models must be in the SAME folder as app.py
yield_model = joblib.load("CRA_Final_Yield_Prediction_Model.joblib")
resilience_model = joblib.load("CRA_Final_Resilient_Classifier_Model.joblib")

# -------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Climate Resilient Agriculture",
    layout="wide"
)

st.title("🌱 Climate Resilient Agriculture – Decision Support System")

st.markdown("""
This application predicts:
- **Crop Yield (kg/ha)** using a climate-aware ML model  
- **Climate Resilience Level** (Low / Medium / High)

The models are trained using **climate, soil, water, and management variables**
to support **sustainable agricultural decision-making**.
""")

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.header("🌦️ Input Parameters")

def get_user_input():
    return pd.DataFrame([{
        "Year": st.sidebar.slider("Year", 2000, 2030, 2020),
        "Avg_Temperature": st.sidebar.slider("Avg Temperature (°C)", 15.0, 35.0, 25.0),
        "Temp_Anomaly": st.sidebar.slider("Temperature Anomaly (°C)", -5.0, 5.0, 0.0),
        "Rainfall_mm": st.sidebar.slider("Rainfall (mm)", 300, 1500, 800),
        "Relative_Humidity": st.sidebar.slider("Relative Humidity (%)", 30, 100, 65),
        "Heatwave_Days": st.sidebar.slider("Heatwave Days", 0, 10, 2),
        "Dry_Spell_Count": st.sidebar.slider("Dry Spell Count", 0, 15, 5),
        "Soil_Organic_Carbon": st.sidebar.slider("Soil Organic Carbon", 0.40, 0.80, 0.55),
        "Soil_pH": st.sidebar.slider("Soil pH", 5.5, 8.5, 6.8),
        "Water_Holding_Capacity": st.sidebar.slider("Water Holding Capacity (%)", 30, 60, 40),
        "Electrical_Conductivity": st.sidebar.slider("Electrical Conductivity", 0.5, 2.0, 1.1),
        "Irrigation_Coverage": st.sidebar.slider("Irrigation Coverage (%)", 0, 60, 30),
        "Groundwater_Depth": st.sidebar.slider("Groundwater Depth (m)", 10, 50, 30),
        "Wind_Speed": st.sidebar.slider("Wind Speed (m/s)", 0.5, 6.0, 2.5),
        "Solar_Radiation": st.sidebar.slider("Solar Radiation", 10.0, 30.0, 20.0),
        "CO2_Concentration": st.sidebar.slider("CO₂ Concentration (ppm)", 380, 480, 420),
        "PM2_5": st.sidebar.slider("PM2.5", 5, 300, 80),
        "PM10": st.sidebar.slider("PM10", 10, 500, 150),
        "Aerosol_Optical_Depth": st.sidebar.slider("Aerosol Optical Depth", 0.05, 3.0, 1.2),
        "Season": st.sidebar.selectbox(
            "Season",
            ["Kharif", "Rabi", "Summer", "Autumn", "Winter", "Whole year"]
        ),
        "State": st.sidebar.selectbox(
            "State",
            ["Maharashtra", "Punjab", "Tamil Nadu", "Andhra Pradesh", "Karnataka"]
        ),
        "District": st.sidebar.selectbox(
            "District",
            ["Pune", "Nagpur", "Chennai", "Amritsar", "Bengaluru"]
        ),
        "Seed_Variety": st.sidebar.selectbox(
            "Seed Variety",
            ["Local", "Hybrid", "HYV", "Traditional"]
        ),
        "Irrigation_Source": st.sidebar.selectbox(
            "Irrigation Source",
            ["Canal", "Rainfed", "Other"]
        )
    }])

input_df = get_user_input()

# -------------------------------------------------
# Climate Stress Index (for resilience model)
# -------------------------------------------------
# Matches training logic conceptually
stress_features = ["Heatwave_Days", "Dry_Spell_Count", "Temp_Anomaly"]

stress_mean = np.array([2.5, 5.0, 0.0])
stress_std  = np.array([2.0, 4.0, 2.0])

stress_values = input_df[stress_features].values.astype(float)

input_df["Climate_Stress_Index"] = (
    (stress_values - stress_mean) / stress_std
).mean(axis=1)

# -------------------------------------------------
# Predictions (PIPELINE HANDLES EVERYTHING)
# -------------------------------------------------
st.subheader("📊 Predictions")

yield_prediction = yield_model.predict(input_df)[0]
resilience_prediction = resilience_model.predict(input_df)[0]

col1, col2 = st.columns(2)

with col1:
    st.metric("🌾 Predicted Yield (kg/ha)", f"{yield_prediction:.0f}")

with col2:
    st.metric("🛡️ Climate Resilience Level", resilience_prediction)

# -------------------------------------------------
# Interpretation
# -------------------------------------------------
st.subheader("🧠 Interpretation")

if resilience_prediction == "High":
    st.success("High resilience: the system can maintain yield under climate stress.")
elif resilience_prediction == "Medium":
    st.warning("Moderate resilience: adaptive measures are recommended.")
else:
    st.error("Low resilience: high vulnerability to climate extremes.")
