# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

model = pickle.load(open("car_price_model.pkl", "rb"))

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class CarFeatures(BaseModel):
    year: int
    mileage: int
    fuel: str
    model: str
    transmission: str

# Encoders
fuel_map = {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'Electric': 3}
model_map = {'Swift': 0, 'Baleno': 1, 'City': 2, 'i20': 3}

# City multipliers
city_multipliers = {
    "Mumbai": 1.10, "Delhi": 1.05, "Hyderabad": 1.08,
    "Chennai": 1.04, "Bangalore": 1.12, "Ranchi": 0.95,
    "Patna": 0.92, "default": 1.0
}

def get_user_city():
    try:
        res = requests.get("https://ipapi.co/json/").json()
        return res.get("city", "default")
    except:
        return "default"

@app.post("/predict")
def predict_price(data: CarFeatures):
    fuel_encoded = fuel_map.get(data.fuel, 0)
    model_encoded = model_map.get(data.model, 0)
    input_array = np.array([data.year, data.mileage, fuel_encoded, model_encoded]).reshape(1, -1)

    base_price = model.predict(input_array)[0]
    city = get_user_city()
    multiplier = city_multipliers.get(city, city_multipliers["default"])
    adjusted_price = round(base_price * multiplier, 2)

    return {
        "base_price": round(base_price, 2),
        "adjusted_price": adjusted_price,
        "city": city,
        "multiplier_used": multiplier
    }
