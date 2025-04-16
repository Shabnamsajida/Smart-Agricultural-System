import streamlit as st
import pandas as pd
import numpy as np
import pickle
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

# Load models
with open("NBClassifier.pkl", "rb") as file:
    nb_model = pickle.load(file)

with open("svr.pkl", "rb") as file:
    svr_model = pickle.load(file)

with open("preprocesser.pkl", "rb") as file:
    preprocesser = pickle.load(file)

with open("smart_irrigation_xgb.pkl", "rb") as file:
    irrigation_model = pickle.load(file)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("yield_df.csv")
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df

df = load_data()

# Crop Emojis Dictionary
crop_emojis = {
    "Rice": "🌾", "Maize": "🌽", "Chickpea": "🫛", "Kidney Beans": "🥣",
    "Pigeon Peas": "🌱", "Moth Beans": "🧀", "Mung Bean": "🌿",
    "Black Gram": "🍛", "Lentil": "🍲", "Pomegranate": "🍎",
    "Banana": "🍌", "Mango": "🧅", "Grapes": "🍇",
    "Watermelon": "🍉", "Muskmelon": "🍈", "Apple": "🍏",
    "Orange": "🍊", "Papaya": "🧅", "Coconut": "🧅",
    "Cotton": "👕", "Jute": "🧅", "Coffee": "☕"
}

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
choice = st.sidebar.radio("Go to", ["Home", "Crop Yield Prediction", "Crop Recommendation", "Smart Irrigation System"])

if choice == "Home":
    st.title("🌾 Welcome to the AI-Powered Agriculture System")
    st.write("""
        **About This Application**
        
        This **AI-powered system** helps farmers and researchers **predict crop yield**, recommend the best crops, and determine **optimal water usage** for irrigation.
        
        **👉 Key Features:**
        - **Crop Yield Prediction** using **Machine Learning Models (SVR)** 📈
        - **Crop Recommendation** based on soil and weather conditions 🌿
        - **Smart Irrigation System** to predict water requirements 💧
        - **Easy-to-Use Streamlit Interface** 🎨
        - **Supports Various Crops and Geographical Areas** 🌎
        - **Optimized for Desktop & Mobile Use** 📱💻
                    """)
    st.write("""
        **About This Application**
        This **AI-powered system** helps farmers and researchers **predict crop yield** based on environmental factors and recommend the best crops for cultivation.
        **🌟 Start Exploring:**
        - Click on **"Crop Prediction"** to get recommendations.
        - Adjust input values to see real-time results.
    """)
    
    # Additional Features
    st.markdown("### 🌱 Features & Benefits")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("✔ **Smart Crop Suggestions** 🌾")
        st.markdown("✔ **Accurate Weather-Based Recommendations** ☁️")
        st.markdown("✔ **Soil Nutrient Analysis** 🧪")
    
    with col2:
        st.markdown("✔ **Supports Multiple Soil Types** 🌎")
        st.markdown("✔ **Easy-to-Use Interface** 🎨")
        st.markdown("✔ **Optimized for All Devices** 📱💻")

elif choice == "Crop Yield Prediction":
    st.title("🌾 Crop Yield Prediction")
    
    Year = st.number_input("Year", min_value=1900, max_value=2100, value=2000)
    average_rain_fall_mm_per_year = st.number_input("Average Rainfall (mm per year)", value=1000.0)
    pesticides_tonnes = st.number_input("Pesticides Used (tonnes)", value=50.0)
    avg_temp = st.number_input("Average Temperature (°C)", value=25.0)
    Area = st.selectbox("Select Area", df['Area'].unique())
    Item = st.selectbox("Select Crop", df['Item'].unique())
    
    if st.button("Predict Yield"):
        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
        transformed_features = preprocesser.transform(features)
        predicted_yield = svr_model.predict(transformed_features).reshape(-1, 1)
        st.success(f"Predicted Yield: {predicted_yield[0][0]}")

elif choice == "Crop Recommendation":
    st.title("🌱 AI-Powered Crop Recommendation System")
    
    Nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=70)
    Phosphorus = st.number_input("Phosphorus (P)", min_value=5, max_value=145, value=75)
    Potassium = st.number_input("Potassium (K)", min_value=5, max_value=205, value=100)
    Temperature = st.number_input("Temperature (°C)", min_value=8, max_value=45, value=25)
    Humidity = st.number_input("Humidity (%)", min_value=10, max_value=100, value=60)
    pH = st.number_input("pH Level", min_value=3.5, max_value=9.5, value=6.5)
    Rainfall = st.number_input("Rainfall (mm)", min_value=20, max_value=300, value=150)
    
    if st.button("Predict Crop"):
        input_array = np.array([[Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall]])
        prediction = nb_model.predict(input_array)[0]
        emoji = crop_emojis.get(prediction, "🌾")
        st.markdown(f"### 🎉 Recommended Crop: {prediction} {emoji}")

elif choice == "Smart Irrigation System":
    st.title("💧 Smart Irrigation System - Water Requirement Predictor")
    
    soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, step=0.1)
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
    crop_type = st.selectbox("Crop Type", ["grapes", "rice", "groundnut", "jute", "maize"])
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, step=0.1)
    sunlight_hours = st.number_input("Sunlight Hours", min_value=0.0, max_value=24.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
    
    crop_mapping = {"grapes": 0, "rice": 1, "groundnut": 2, "jute": 3, "maize": 4}
    crop_encoded = crop_mapping[crop_type]
    
    if st.button("Predict Water Requirement"):
        input_data = np.array([[soil_moisture, temperature, crop_encoded, wind_speed, sunlight_hours, humidity]])
        prediction = irrigation_model.predict(input_data)
        st.success(f"💧 Estimated Water Requirement: {prediction[0]:.2f} liters")
