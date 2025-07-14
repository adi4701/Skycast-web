# %% [markdown]
# Importing the dependencies

# %%
import requests # Fetches data from the api
import pandas as pd # Data handling and analysis
import numpy as np # Numerical operations
from sklearn.model_selection import train_test_split # Splitting data into training and testing sets
from sklearn.preprocessing import LabelEncoder # categorical data to numerical values
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # model for classification and regression
from sklearn.metrics import accuracy_score, mean_squared_error # measures the accuracy of classification and regression models
from datetime import datetime,timedelta # for handling date and time
import pytz # for timezone handling

# %%
API_KEY = 'ca235d315a2737cb3e83d116df42fe3f'
BASE_URL = 'https://api.openweathermap.org/data/2.5/weather?lat=44.34&lon=10.99&appid=ca235d315a2737cb3e83d116df42fe3f'

# %% [markdown]
# Fetching Current Weather Data #could be an error here because of ? before q
# 

# %%
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric" 
    response = requests.get(url)
    data = response.json()
    return {
        'city': data['name'],
        'current_temperature': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temperature_min': round(data['main']['temp_min']),
        'temperature_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'WindGustDir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'WindGustSpeed': round(data['wind']['speed'] * 3.6),  # Convert m/s to km/h
    }   

# %% [markdown]
# Historical Data 

# %%
def read_historical_weather_data(filename):
    df = pd.read_csv(filename) #load csv file into dataframe
    df = df.dropna() #missing values are dropped
    df = df.drop_duplicates() #duplicate rows are dropped
    return df

# %% [markdown]
# Data for Training
# 
# 

# %%
def prepare_data(data):
    le = LabelEncoder() # create an instance of LabelEncoder
    data['WindGustDir'] = le.fit_transform(data['WindGustDir']) 
    data['RainTommorow'] = le.fit_transform(data['RainTommorow']) 

    #define feature and target variables
    X = data[['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'Humidity', 'WindGustDir']]
    y = data['RainTommorow']

    return X, y 

# %% [markdown]
# Train Rain Prediction Model
# 

# %%
def train_rain_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42) #split data into training and testing sets
    model = RandomForestClassifier(n_estimators=100, random_state=42) #create a random forest classifier
    model.fit(X_train, y_train) #fit the model on training data

    y_pred = model.predict(X_test) #make predictions on the test data

    print("mean squared error:", mean_squared_error(y_test, y_pred)) #calculate the mean squared error
    
    
    return model #return the trained model

# %% [markdown]
# Regression Data preparation
# 

# %%
def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])
    
    X = np.array(X).reshape(-1, 1)  # Reshape for regression
    y = np.array(y)
    return X, y

# %% [markdown]
# Train Regression Data

# %%
def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators = 100, random_state = 42)  # create a random forest regressor
    model.fit(X, y)
    return model

# %% [markdown]
# Predict Future

# %%
def predict_future(model, current_value):
    predictions = [current_value]
    for i in range(7):
        next_value = model.predict(np.array(predictions[-1]).reshape(-1, 1))

        predictions.append(next_value[0])
    return predictions[1:]  # return the next 7 days predictions


# %% [markdown]
# Weather Analysis Function

# %%
def weather_view():
    print("Welcome to the Weather Analysis App!")
    print("This app provides current weather information and rainfall predictions.")
    city = input("Enter the city name: ")
    current_weather = get_current_weather(city)
    print(f"Current weather in {current_weather['city']}, {current_weather['country']}:")

    #load historical weather data
    historical_data = read_historical_weather_data('C:/Sanskar/Python(VS)/Rainfall Prediction/weather.csv')
    print("Historical weather data loaded successfully.")

    #prepare and train the rainfall prediction model
    
    X, y, le = prepare_data(historical_data)
    rain_model = train_rain_model(X, y)

    print("Rainfall prediction model trained successfully.")

    #map wind direction to compass points
    wind_degree = current_weather['wind_gust_dir']%360
    compass_points = [
        ("N",0,11.25), ("NNE",11.25,33.75), ("NE",33.75,56.25),
        ("ENE",56.25,78.75), ("E",78.75,101.25), ("ESE",101.25,123.75),
        ("SE",123.75,146.25), ("SSE",146.25,168.75), ("S",168.75,191.25),
        ("SSW",191.25,213.75), ("SW",213.75,236.25), ("WSW",236.25,258.75),
        ("W",258.75,281.25), ("WNW",281.25,303.75), ("NW",303.75,326.25),
        ("NNW",326.25,348.75), ("N",348.75,360)
    ]

    compass_direction = next(point for point,start, end in compass_points if start <= wind_degree < end)

    compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

    current_data = {
        'MinTemp': current_weather['temperature_min'],
        'MaxTemp': current_weather['temperature_max'],
        'WindGustDir': compass_direction_encoded, 
        'WindGustSpeed': current_weather['wind_gust_speed'],
        'humidity': current_weather['humidity'],
        'pressure': current_weather['pressure'],
        'Temperature': current_weather['current_temperature'],
        'Rainfall': 0,  # Assuming no rainfall data for current weather
    }

    current_df = pd.DataFrame([current_data])

    #Rain prediction

    rain_prediction = rain_model.predict(current_df)[0]

    #prepare regression data for temperature and humidity

    X_temp, y_temp = prepare_regression_data(historical_data, 'Temperature')
    X_hum, y_hum = prepare_regression_data(historical_data, 'humidity')

    temp_model = train_regression_model(X_temp, y_temp)
    temp_model = train_regression_model(X_hum, y_hum)

    #predict future temperature and humidity
    future_temp = predict_future(temp_model, current_data['Temperature'])
    future_hum = predict_future(temp_model, current_data['humidity'])
   
   #prepare time for future predictions

    timezone = pytz.timezone('Asia/Kolkata')  # Set your desired timezone
    now = datetime.now(timezone)
    next_hour = now + timedelta(hours=1)
    next_hour = next_hour.replace(minute=0, second=0, microsecond=0)  # Round to the next hour

    future_times = [next_hour + timedelta(hours=i) for i in range(7)]

    future_dates = [(now + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]


    #Results

    print(f"City: {city}, {current_weather['country']} ")
    print(f"Current Temperature: {current_weather['current_temperature']}°C")
    print(f"Feels Like: {current_weather['feels_like']}°C")
    print(f"Min Temperature: {current_weather['temperature_min']}°C")
    print(f"Max Temperature: {current_weather['temperature_max']}°C")
    print(f"Humidity: {current_weather['humidity']}%")
    print(f"Wind Gust Direction: {compass_direction} ({current_weather['WindGustDir']}°)")
    print(f"Wind Gust Speed: {current_weather['WindGustSpeed']} km/h")
    print(f"Pressure: {current_weather['pressure']} hPa")
    print(f"Weather Description: {current_weather['description']}")
    print(f"Rainfall Prediction for Tomorrow: {'Yes' if rain_prediction == 1 else 'No'}")


    print("/nFuture Predictions:")

    for time, temp in zip(future_times, future_temp):
        print(f"{time}: {round(temp,1)}°C")

    print("/nFuture Humidity Predictions:")

    for time,humidity in zip(future_times, future_hum):
        print(f"{time}: {round(humidity,1)}%")

weather_view()


