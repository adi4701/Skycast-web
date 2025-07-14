from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import requests

app = Flask(__name__)

# Load models
HUM_MODEL_PATH = 'humidity_model.pkl'
RAIN_MODEL_PATH = 'rainfall_model.pkl'
WEATHER_CSV_PATH = '../weather.csv'
API_KEY = 'ca235d315a2737cb3e83d116df42fe3f'  # OpenWeatherMap API key
CITY = 'Gurgaon,IN'

@app.route('/')
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        humidity = float(data['humidity'])
        pressure = float(data['pressure'])
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'Invalid input. Provide humidity and pressure as numbers.'}), 400
    model = joblib.load('model.pkl')
    features = np.array([[humidity, pressure]])
    predicted_temp = model.predict(features)[0]
    return jsonify({'predicted_temperature': round(float(predicted_temp), 2)})

@app.route('/forecast', methods=['GET'])
def forecast():
    # --- Real-time data from OpenWeatherMap ---
    try:
        url = f'http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            realtime_humidity = data['main']['humidity']
            realtime_temp = data['main']['temp']
            realtime_pressure = data['main']['pressure']
            realtime_wind = round(data['wind'].get('speed', 0) * 3.6)
            # Rainfall: OpenWeatherMap gives rain volume, not probability. We'll show 1 if rain in last hour, else 0.
            realtime_rain = 1 if 'rain' in data and data['rain'].get('1h', 0) > 0 else 0
        else:
            realtime_humidity = None
            realtime_temp = None
            realtime_pressure = None
            realtime_wind = None
            realtime_rain = None
    except Exception:
        realtime_humidity = None
        realtime_temp = None
        realtime_pressure = None
        realtime_wind = None
        realtime_rain = None

    # --- Model-based 7-day forecast ---
    df = pd.read_csv(WEATHER_CSV_PATH)
    df = df.dropna()
    df = df.drop_duplicates()
    last_n = 90
    last_n_df = df.tail(last_n).reset_index(drop=True)
    le = LabelEncoder()
    last_n_df['RainTomorrow'] = le.fit_transform(last_n_df['RainTomorrow'])
    feature_cols = ['Humidity', 'Temp', 'Pressure', 'WindGustSpeed']
    window = 7
    hum_model = joblib.load(HUM_MODEL_PATH)
    rain_model = joblib.load(RAIN_MODEL_PATH)
    # Use last 6 days from CSV, most recent day from real-time
    feats = {col: last_n_df[col].iloc[-(window-1):].tolist() for col in feature_cols}
    for idx, col in enumerate(feature_cols):
        if col == 'Humidity':
            feats[col].append(realtime_humidity)
        elif col == 'Temp':
            feats[col].append(realtime_temp)
        elif col == 'Pressure':
            feats[col].append(realtime_pressure)
        elif col == 'WindGustSpeed':
            feats[col].append(realtime_wind)
    # For rainfall, use last 6 days from CSV, and real-time rain (0/1) for today
    rain_input = last_n_df['RainTomorrow'].iloc[-(window-1):].tolist()
    rain_input.append(realtime_rain if realtime_rain is not None else 0)
    hum_preds = []
    rain_preds = []
    rain_probs = []
    high_rain_days = []
    for i in range(7):
        # Prepare input for this step
        hum_features = []
        for col in feature_cols:
            hum_features.extend(feats[col][-window:])
        hum_pred = hum_model.predict([hum_features])[0]
        hum_preds.append(round(float(hum_pred), 2))
        rain_features = []
        for col in feature_cols:
            rain_features.extend(feats[col][-window:])
        rain_proba = rain_model.predict_proba([rain_features])[0]
        rain_prob = round(float(rain_proba[1]) * 100, 1)
        rain_probs.append(rain_prob)
        rain_pred = rain_model.predict([rain_features])[0]
        rain_preds.append(int(rain_pred))
        high_rain_days.append(rain_prob > 60)
        # Roll window: append prediction for next day
        feats['Humidity'].append(hum_pred)
        # For other features, just repeat last value (or could use model, but keep simple)
        for col in ['Temp', 'Pressure', 'WindGustSpeed']:
            feats[col].append(feats[col][-1])
        rain_input.append(rain_pred)
    today = datetime.now()
    dates = [(today + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(7)]
    rain_labels = ['Yes' if r == 1 else 'No' for r in rain_preds]
    return jsonify({
        'dates': dates,
        'humidity': hum_preds,
        'rainfall': rain_labels,
        'rainfall_probabilities': rain_probs,
        'high_rain_days': high_rain_days,
        'realtime_humidity': realtime_humidity,
        'realtime_temp': realtime_temp,
        'realtime_rain': 'Yes' if realtime_rain == 1 else 'No' if realtime_rain == 0 else None
    })

if __name__ == '__main__':
    app.run(debug=True) 