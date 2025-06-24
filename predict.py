import pickle
import numpy as np

def load_model_and_maps():
    with open('model/car_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/label_maps.pkl', 'rb') as f:
        maps = pickle.load(f)
    return model, maps

def predict_price(year, mileage, fuel, model_name, transmission):
    model, maps = load_model_and_maps()

    fuel_idx = maps['fuel_map'].get(fuel)
    model_idx = maps['model_map'].get(model_name)
    trans_idx = maps['transmission_map'].get(transmission)

    if None in (fuel_idx, model_idx, trans_idx):
        raise ValueError("Invalid fuel, model, or transmission type.")

    input_data = np.array([[year, mileage, fuel_idx, model_idx, trans_idx]])
    predicted_price = model.predict(input_data)[0]
    return round(predicted_price, 2)
