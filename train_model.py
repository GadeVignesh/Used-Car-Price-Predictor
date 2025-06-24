import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Expanded dataset
data = {
    'year': [
        2015, 2018, 2020, 2017, 2016, 2019, 2013, 2021, 2022, 2014,
        2016, 2017, 2018, 2019, 2020, 2020, 2021, 2022, 2023, 2015,
        2017, 2018, 2019, 2020, 2021, 2016, 2017, 2018, 2019, 2022,
        2018, 2019, 2020, 2021, 2022, 2023, 2017, 2016, 2015, 2019,
        2018, 2020, 2021, 2023, 2022, 2020, 2019, 2018, 2021, 2023
    ],
    'mileage': [
        60000, 40000, 30000, 50000, 70000, 20000, 85000, 15000, 10000, 90000,
        75000, 65000, 35000, 30000, 22000, 15000, 10000, 5000, 2000, 80000,
        68000, 55000, 42000, 29000, 12000, 76000, 54000, 47000, 26000, 10000,
        48000, 35000, 22000, 12000, 5000, 3000, 60000, 75000, 90000, 20000,
        25000, 18000, 11000, 3000, 7000, 22000, 34000, 40000, 15000, 8000
    ],
    'fuel': [
        'Petrol', 'Diesel', 'Petrol', 'CNG', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Electric', 'Diesel',
        'Petrol', 'Diesel', 'Petrol', 'Diesel', 'CNG', 'Petrol', 'Diesel', 'Electric', 'Petrol', 'Diesel',
        'Diesel', 'Petrol', 'CNG', 'Electric', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'CNG',
        'Petrol', 'Diesel', 'Petrol', 'Electric', 'Petrol', 'CNG', 'Diesel', 'Petrol', 'Diesel', 'Petrol',
        'Petrol', 'Petrol', 'Diesel', 'Electric', 'Diesel', 'Petrol', 'Diesel', 'CNG', 'Petrol', 'Electric'
    ],
    'model': [
        'Swift', 'Baleno', 'City', 'i20', 'i20', 'Swift', 'City', 'Baleno', 'Nexon', 'City',
        'i20', 'Swift', 'Baleno', 'City', 'i20', 'Swift', 'Baleno', 'City', 'Altroz', 'i20',
        'Creta', 'Verna', 'Tigor', 'Tiago', 'Seltos', 'XUV300', 'Innova', 'Venue', 'Fronx', 'Punch',
        'Glanza', 'Ciaz', 'Fortuner', 'MG Hector', 'Kiger', 'Sonet', 'XUV700', 'Ertiga', 'EcoSport', 'Jazz',
        'Amaze', 'i10', 'Aura', 'Zest', 'Harrier', 'Scorpio', 'KUV100', 'Kicks', 'RediGO', 'Comet EV'
    ],
    'transmission': [
        'Manual', 'Automatic', 'Manual', 'Manual', 'Automatic', 'Manual', 'Automatic', 'Manual', 'Automatic', 'Manual',
        'Manual', 'Automatic', 'Manual', 'Manual', 'Manual', 'Manual', 'Automatic', 'Automatic', 'Manual', 'Manual',
        'Automatic', 'Manual', 'Manual', 'Automatic', 'Manual', 'Manual', 'Manual', 'Automatic', 'Manual', 'Manual',
        'Manual', 'Manual', 'Automatic', 'Automatic', 'Manual', 'Manual', 'Automatic', 'Manual', 'Manual', 'Manual',
        'Manual', 'Manual', 'Manual', 'Automatic', 'Manual', 'Manual', 'Manual', 'Manual', 'Manual', 'Automatic'
    ],
    'price': [
        350000, 500000, 700000, 400000, 450000, 375000, 320000, 610000, 780000, 300000,
        425000, 360000, 540000, 660000, 390000, 410000, 570000, 950000, 620000, 340000,
        890000, 670000, 450000, 600000, 950000, 870000, 1000000, 740000, 820000, 630000,
        585000, 525000, 2100000, 1750000, 740000, 710000, 1900000, 830000, 580000, 490000,
        470000, 350000, 465000, 970000, 1320000, 900000, 435000, 510000, 370000, 1100000
    ]
}

df = pd.DataFrame(data)

# Encoding maps
fuel_map = {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'Electric': 3}
model_list = sorted(list(set(data['model'])))
model_map = {model_name: idx for idx, model_name in enumerate(model_list)}
transmission_map = {'Manual': 0, 'Automatic': 1}

# Apply encoding
df['fuel'] = df['fuel'].map(fuel_map)
df['model'] = df['model'].map(model_map)
df['transmission'] = df['transmission'].map(transmission_map)

# Drop rows with NaN (from unmapped values, just in case)
df.dropna(inplace=True)

# Ensure integer types after dropping NaNs
df = df.astype({'fuel': int, 'model': int, 'transmission': int})

# Features and target
X = df[['year', 'mileage', 'fuel', 'model', 'transmission']]
y = df['price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs('model', exist_ok=True)

with open('model/car_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/label_maps.pkl', 'wb') as f:
    pickle.dump({
        'fuel_map': fuel_map,
        'model_map': model_map,
        'transmission_map': transmission_map
    }, f)

print("✅ Model trained on", len(df), "cars and saved to 'model/' folder.")
