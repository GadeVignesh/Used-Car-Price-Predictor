## USED CAR PRICE PREDICTION APP

Overview:
This is a Python + Streamlit-based application that predicts the resale price of used cars.
The prediction is based on parameters like car model, year of purchase, mileage, fuel type, and transmission type.
The machine learning model used is a Random Forest Regressor.

---

Features:

- Predict resale value of used cars using AI
- Inputs: Model, Year, Mileage, Fuel Type, Transmission
- Returns estimated price and ±10% confidence range
- Simple Streamlit web interface
- Trained on real-world dataset
- Ready for local or cloud deployment

---

Project Structure:

car-price-predictor/
│
├── app.py -> Streamlit frontend (UI)
├── train_model.py -> Model training script
├── model/
│ ├── car_price_model.pkl -> Trained ML model
│ └── label_maps.pkl -> Encoded label mappings
├── requirements.txt -> Python dependencies
└── README.txt -> Project documentation (you are here)

---

Tech Stack:

- Python 3.10+
- Streamlit
- scikit-learn
- pandas
- numpy

---

How to Run:

1. Clone the project:
   git clone https://github.com/yourusername/car-price-predictor
   cd car-price-predictor

2. Install dependencies:
   pip install -r requirements.txt

3. Train the model:
   python train_model.py

4. Run the Streamlit app:
   streamlit run app.py

---

Deployment:
You can deploy this project on:

- Streamlit Cloud
- Render
- Heroku
- Localhost for testing

---

Author:
Gade Vignesh
LinkedIn: https://www.linkedin.com/in/vignesh-gade/

---

License:
MIT License – free to use, modify, and distribute.
