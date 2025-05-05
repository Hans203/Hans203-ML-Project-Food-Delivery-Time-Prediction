import streamlit as st
import pandas as pd
import pickle

# Load both models
with open("best_rf_model.pkl", "rb") as rf_file:
    rf_model = pickle.load(rf_file)

with open("best_xgb_model.pkl", "rb") as xgb_file:
    xgb_model = pickle.load(xgb_file)

# Streamlit UI
st.set_page_config(page_title="Delivery Time Predictor", layout="centered")
st.title("🚚 Delivery Time Predictor")
st.markdown("Predict estimated delivery time using either **Random Forest** or **XGBoost**.")

# --- Model selector ---
model_choice = st.selectbox("Choose a model:", ["Random Forest", "XGBoost"])
model = rf_model if model_choice == "Random Forest" else xgb_model

# --- Input fields ---
traffic_level = st.selectbox("Traffic Level", ["Low", "Moderate", "High"])
distance = st.slider("Distance (km)", 1.0, 20.0, 5.0, 0.1)

# --- Age Input ---
age = st.number_input("Delivery Person Age", min_value=18, max_value=65, value=30)
    
# --- Weather Description Section ---
st.markdown("<h5 style='text-align: left;'>Weather Description</h5>", unsafe_allow_html=True)
weather_desc = st.selectbox("Type of Weather", ["Clear", "Cloudy", "Hazy/Foggy"])
temperature = st.slider("Temperature (°C)", 10.0, 45.0, 30.0, 0.5)
humidity = st.slider("Humidity (%)", 10, 100, 50, 1)

# --- Order Section ---
order_type = st.selectbox("Type of Order", ["Buffet", "Drinks", "Meal", "Snack"])




# --- Encode inputs ---
traffic_map = {"Low": 0, "Moderate": 1, "High": 2}
type_meal = 1 if order_type == 'Meal' else 0
type_drinks = 1 if order_type == 'Drinks' else 0
type_snack = 1 if order_type == 'Snack' else 0

# Weather one-hot encoding (drop_first=True logic: Clear = baseline)
weather_cloudy = 1 if weather_desc == "Cloudy" else 0
weather_hazy = 1 if weather_desc == "Hazy/Foggy" else 0

# --- Create Input DataFrame ---
input_data = pd.DataFrame([{
    'Traffic_Level_Encoded': traffic_map[traffic_level],
    'Distance (km)': distance,
    'Delivery_person_Age': age,
    'weather_description_Hazy/Foggy': weather_hazy,
    'weather_description_Cloudy': weather_cloudy,
    'temperature': temperature,
    'humidity': humidity,
    
    'Type_of_order_Drinks': type_drinks,
    'Type_of_order_Meal': type_meal,
    'Type_of_order_Snack': type_snack
}])

# --- Predict button ---
if st.button("Predict Delivery Time"):
    try:
        prediction = model.predict(input_data)[0]
        formatted_pred = f"{prediction:.2f}"
        st.success(f"Using {model_choice}: Estimated Delivery Time is **{formatted_pred} minutes**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Built with Streamlit, XGBoost, and Random Forest")
