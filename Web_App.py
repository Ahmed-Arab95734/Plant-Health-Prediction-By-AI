import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# 🔧 CONFIGURATION
# =========================
# Paste your company logo URL here
COMPANY_LOGO_URL = "https://admin.arcom.com.eg/images/page/6792191f561a3.png" # Placeholder plant icon

st.set_page_config(
    page_title="Plant Health Monitoring",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =========================
# 🎨 CUSTOM CSS (Fonts & Styling)
# =========================
# This block handles the "Bigger Fonts" and Background
page_style = """
<style>
    /* Main app background */
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.pexels.com/photos/1072824/pexels-photo-1072824.jpeg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: #ffffff;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: rgba(20, 50, 20, 0.95);
        color: #ffffff;
    }

    /* --- FONT SIZE ADJUSTMENTS --- */
    
    /* Main Title (H1) */
    h1 {
        font-size: 2.3rem !important; /* Much bigger */
        font-weight: 700 !important;
        color: #ffffff !important;
        text-shadow: 3px 3px 6px #000000;
    }

    /* Subheaders (H2, H3) */
    h2, h3 {
        font-size: 2.0rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        text-shadow: 2px 2px 4px #000000;
    }

    /* Standard Text (Paragraphs) */
    p, li, .stMarkdown {
        font-size: 1.8rem !important; /* Bigger body text */
        font-weight: 630 !important;
        color: #f0f0f0 !important;
        text-shadow: 1px 1px 2px #000000;
        line-height: 1.6 !important;
    }

    /* Sidebar Labels */
    .stSlider label, .stSelectbox label {
        font-size: 1.4rem !important;
        color: #ffffff !important;
        font-weight: 630 !important;
    }

    /* Sidebar Values */
    div[data-testid="stMarkdownContainer"] p {
        font-size: 1.4rem !important;
    }
    
    /* Success/Prediction Box Text */
    .prediction-box h2 {
        font-size: 2.5rem !important;
        text-shadow: none;
    }

</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# =========================
# 🏗️ HEADER WITH LOGO
# =========================
# Using columns to put Title on left and Logo on right
col_header, col_logo = st.columns([4, 1])

with col_header:
    st.title("🌿 Plant Health Prediction By AI")

with col_logo:
    st.image(COMPANY_LOGO_URL, width=120) 



st.markdown(
    """
    This interactive web app predicts the health status of a plant based on environmental 
    and soil conditions using an Artificial Intelligence model.
    
    The model classifies the plant into one of three states:
    * **Healthy** (Optimal conditions)
    * **Moderate Stress** (Requires attention)
    * **High Stress** (Critical condition)
    """
)

st.markdown("---")
st.subheader("📊 About The Data")
st.write("The model analyzes factors like Soil Moisture, Temperature, Light Intensity, and Nutrient Levels to determine plant health.")




# =========================
# 📥 LOAD MODEL
# =========================
try:
    model = joblib.load("plant_health_model.pkl")
except FileNotFoundError:
    st.error("⚠️ Model file 'plant_health_model.pkl' not found. Please ensure you have saved the model.")
    st.stop()

# Define labels and colors
health_labels = {0: "Healthy", 1: "Moderate Stress", 2: "High Stress"}
health_colors = {0: "#4CAF50", 1: "#FFC107", 2: "#FF5252"} 

# =========================
# ⚙️ SIDEBAR INPUTS
# =========================
st.sidebar.header("🔧 Input Conditions")

Soil_Moisture = st.sidebar.slider("Soil Moisture (%)", 10.0, 40.0, 30.5)
Ambient_Temperature = st.sidebar.slider("Ambient Temperature (°C)", 18.0, 30.0, 21.88)
Soil_Temperature = st.sidebar.slider("Soil Temperature (°C)", 15.0, 25.0, 17.38)
Humidity = st.sidebar.slider("Humidity (%)", 40.0, 70.0, 53.64)
Light_Intensity = st.sidebar.slider("Light Intensity", 200.0, 1000.0, 418.43)
Soil_pH = st.sidebar.slider("Soil pH", 5.5, 7.5, 6.92)
Nitrogen_Level = st.sidebar.slider("Nitrogen Level", 10.0, 50.0, 28.99)
Phosphorus_Level = st.sidebar.slider("Phosphorus Level", 10.0, 50.0, 25.16)
Potassium_Level = st.sidebar.slider("Potassium Level", 10.0, 50.0, 36.05)
Chlorophyll_Content = st.sidebar.slider("Chlorophyll Content", 20.0, 50.0, 43.32)
Electrochemical_Signal = st.sidebar.slider("Electrochemical Signal", 0.0, 2.0, 1.3)

# Input array (Order must match training!)
input_data = np.array([[
    Soil_Moisture, Ambient_Temperature, Soil_Temperature, Humidity, 
    Light_Intensity, Soil_pH, Nitrogen_Level, Phosphorus_Level, 
    Potassium_Level, Chlorophyll_Content, Electrochemical_Signal
]])

# =========================
# 🤖 PREDICTION
# =========================
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0]

# =========================
# Show Results
# =========================
st.markdown("---")
st.subheader("📌 Prediction Result")

# Dynamic status box color
status_color = health_colors[prediction]
st.markdown(
    f"""
    <div style="background-color: {status_color}; padding: 20px; border-radius: 10px; text-align: center;">
        <h2 style="color: white; margin: 0;">{health_labels[prediction]}</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# Probabilities + Chart Side by Side
# =========================
st.subheader("📊 Confidence Levels")

col1, col2 = st.columns([2, 1])  # wider column for text, smaller for chart

with col1:
    st.write(f"🟢 **Healthy:** {prediction_proba[0]:.2%}")
    st.write(f"🟠 **Moderate Stress:** {prediction_proba[1]:.2%}")
    st.write(f"🔴 **High Stress:** {prediction_proba[2]:.2%}")

with col2:
    fig, ax = plt.subplots(figsize=(2, 2))  # smaller chart
    labels = ["Healthy", "Moderate", "High"]
    colors = ["#4CAF50", "#FFC107", "#FF5252"] # Green, Amber, Red
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(prediction_proba, labels=None, autopct="%1.1f%%",
           startangle=90, colors=colors, textprops={"fontsize": 8})
    ax.axis("equal")
    
    # Add legend to avoid clutter on small chart
    ax.legend(wedges, labels, title="Status", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=6)

    st.pyplot(fig)