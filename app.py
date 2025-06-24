import streamlit as st
import pickle
import numpy as np

# ✅ This must be the first Streamlit command
st.set_page_config(page_title="Resale Radar", page_icon="🚗")

# Load model and encoders
@st.cache_resource
def load_model_and_maps():
    with open('model/car_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/label_maps.pkl', 'rb') as f:
        maps = pickle.load(f)
    return model, maps

# Load model and label maps
model, maps = load_model_and_maps()

# UI Title and Description
st.title("🚗 Resale Radar - Used Car Price Estimator")
st.markdown("Enter your car details below to get the **estimated resale price** with helpful insights.")

# Sidebar inputs
st.sidebar.header("Car Specifications")
year = st.sidebar.number_input("Year of Purchase", min_value=2000, max_value=2025, step=1, value=2020)
mileage = st.sidebar.number_input("Mileage (in KM)", min_value=0, max_value=300000, step=1000, value=25000)
fuel = st.sidebar.selectbox("Fuel Type", list(maps['fuel_map'].keys()))
model_name = st.sidebar.selectbox("Car Model", list(maps['model_map'].keys()))
transmission = st.sidebar.selectbox("Transmission Type", list(maps['transmission_map'].keys()))

# Predict on button click
if st.sidebar.button("Predict Price"):
    try:
        input_vector = np.array([[
            year,
            mileage,
            maps['fuel_map'][fuel],
            maps['model_map'][model_name],
            maps['transmission_map'][transmission]
        ]])
        predicted_price = model.predict(input_vector)[0]
        lower_bound = predicted_price * 0.90
        upper_bound = predicted_price * 1.10

        # Output Section
        st.subheader("📋 Car Details")
        st.write(f"**Model:** {model_name}")
        st.write(f"**Year:** {year}")
        st.write(f"**Mileage:** {mileage:,} km")
        st.write(f"**Fuel Type:** {fuel}")
        st.write(f"**Transmission:** {transmission}")

        st.subheader("💰 Estimated Resale Price")
        st.success(f"₹{int(predicted_price):,} (±10%)")
        st.write(f"Expected Range: ₹{int(lower_bound):,} – ₹{int(upper_bound):,}")

        st.markdown("---")
        st.caption("📌 Note: This is an AI-generated estimate. Real-world prices may vary based on market conditions and car condition.")

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
