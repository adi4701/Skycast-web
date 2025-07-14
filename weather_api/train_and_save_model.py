import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load the historical weather data
# Adjust the path if running from a different directory
weather_df = pd.read_csv('../weather.csv')

# Clean and prepare data
weather_df = weather_df.dropna()
weather_df = weather_df.drop_duplicates()

# Use Humidity and Pressure as features to predict Temp (temperature)
X = weather_df[['Humidity', 'Pressure']]
y = weather_df['Temp']

# Split data (optional, for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate (optional)
y_pred = model.predict(X_test)
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))

# Save the model
joblib.dump(model, 'model.pkl')
print('Model saved as model.pkl') 