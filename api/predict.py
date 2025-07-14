import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import requests

app = Flask(__name__)

HUM_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../weather_api/humidity_model.pkl')
RAIN_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../weather_api/rainfall_model.pkl')
WEATHER_CSV_PATH = os.path.join(os.path.dirname(__file__), '../weather.csv')
API_KEY = 'ca235d315a2737cb3e83d116df42fe3f'
CITY = 'Gurgaon,IN'

@app.route('/predict', methods=['GET'])
def predict():
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
    rain_input = last_n_df['RainTomorrow'].iloc[-(window-1):].tolist()
    rain_input.append(realtime_rain if realtime_rain is not None else 0)
    hum_preds = []
    rain_preds = []
    rain_probs = []
    high_rain_days = []
    for i in range(7):
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
        feats['Humidity'].append(hum_pred)
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
        'realtime_wind': realtime_wind,
        'realtime_rain': 'Yes' if realtime_rain == 1 else 'No' if realtime_rain == 0 else None
    })

# Vercel entrypoint
def handler(event, context):
    from werkzeug.middleware.dispatcher import DispatcherMiddleware
    from werkzeug.wrappers import Response
    return app(event, context)

if __name__ == '__main__':
    app.run(debug=True) 