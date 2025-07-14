import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
weather_df = pd.read_csv('../weather.csv')
weather_df = weather_df.dropna()
weather_df = weather_df.drop_duplicates()

# Use last 90 days for more robust training
last_n = 90
last_n_df = weather_df.tail(last_n).reset_index(drop=True)

# Encode RainTomorrow as 0/1
le = LabelEncoder()
last_n_df['RainTomorrow'] = le.fit_transform(last_n_df['RainTomorrow'])

# Features to use
feature_cols = ['Humidity', 'Temp', 'Pressure', 'WindGustSpeed']
window = 7

# --- Humidity Forecast Model (Regression) ---
X_hum = []
y_hum = []
for i in range(len(last_n_df) - window):
    # Use windowed features for each feature
    window_feats = []
    for col in feature_cols:
        window_feats.extend(last_n_df[col].iloc[i:i+window].values)
    X_hum.append(window_feats)
    y_hum.append(last_n_df['Humidity'].iloc[i+window])
X_hum = np.array(X_hum)
y_hum = np.array(y_hum)

hum_model = RandomForestRegressor(n_estimators=100, random_state=42)
hum_model.fit(X_hum, y_hum)
joblib.dump(hum_model, 'humidity_model.pkl')

# --- Rainfall Forecast Model (Classification) ---
X_rain = []
y_rain = []
for i in range(len(last_n_df) - window):
    window_feats = []
    for col in feature_cols:
        window_feats.extend(last_n_df[col].iloc[i:i+window].values)
    X_rain.append(window_feats)
    y_rain.append(last_n_df['RainTomorrow'].iloc[i+window])
X_rain = np.array(X_rain)
y_rain = np.array(y_rain)

rain_model = RandomForestClassifier(n_estimators=100, random_state=42)
rain_model.fit(X_rain, y_rain)
joblib.dump(rain_model, 'rainfall_model.pkl')

print('Models trained and saved: humidity_model.pkl, rainfall_model.pkl') 