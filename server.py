from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load model & encoder & columns
model = pickle.load(open('models/regression.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    # 🔹 Get inputs
    present_price = float(request.form.get('present_price'))
    kms = int(request.form.get('kms_driven'))
    age = int(request.form.get('age'))
    fuel = request.form.get('fuel_type')

    fuel_int = 0

    if fuel == 'Petrol':
        fuel_int = 0
    elif fuel == 'Diesel':
        fuel_int = 1
    
    scaled_input = scaler.transform([[present_price,kms,fuel_int,age]])

    predict = model.predict(scaled_input)

    return render_template('index.html', prediction = f"The price is {predict[0]:.2f} lacks")


if __name__ == '__main__':
    app.run(debug=True)