import requests
import pandas as pd
from datetime import datetime
import os
import sys
import joblib
from train_forecast_models import *  # reuse model training logic if possible

# --- CONFIG ---
API_KEY = 'ca235d315a2737cb3e83d116df42fe3f'  # Replace with your OpenWeatherMap API key
CITY = 'Gurgaon,IN'
CSV_PATH = '../weather.csv'

# --- Fetch real-time weather data ---
def fetch_gurgaon_weather():
    url = f'http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric'
    response = requests.get(url)
    if response.status_code != 200:
        print('Failed to fetch weather data:', response.text)
        sys.exit(1)
    data = response.json()
    # Map API fields to CSV columns
    row = {
        'MinTemp': data['main']['temp_min'],
        'MaxTemp': data['main']['temp_max'],
        'WindGustDir': data['wind'].get('deg', 0),
        'WindGustSpeed': round(data['wind'].get('speed', 0) * 3.6),
        'Humidity': data['main']['humidity'],
        'Pressure': data['main']['pressure'],
        'Temp': data['main']['temp'],
        'RainTomorrow': 'No'  # Real-time API can't know tomorrow's rain, so default to 'No'
    }
    return row

# --- Append to CSV if today's data not present ---
def append_if_new(row):
    df = pd.read_csv(CSV_PATH)
    today = datetime.now().strftime('%Y-%m-%d')
    # Check if today's data is present (by Temp and Humidity and MaxTemp)
    if not ((df['Temp'] == row['Temp']) & (df['Humidity'] == row['Humidity']) & (df['MaxTemp'] == row['MaxTemp'])).any():
        df = df.append(row, ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        print('Appended new real-time data to weather.csv')
    else:
        print('Today\'s data already present in weather.csv')

if __name__ == '__main__':
    row = fetch_gurgaon_weather()
    append_if_new(row)
    # Retrain models
    print('Retraining models...')
    os.system('python train_forecast_models.py')
    print('Done.') 