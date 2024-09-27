import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, CORSMiddleware
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import csv
from io import StringIO
import uvicorn
from alert import *

# Load environment variables
load_dotenv()

app = FastAPI()
background_tasks = BackgroundTasks()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")

# Function to get weather data
def get_weather_data(city: str, days: int = 30):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/next{days}days"
    params = {
        "unitGroup": "metric",
        "key": API_KEY,
        "include": "days"
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error fetching weather data")
    
    return response.json()

# Data Preprocessing with Pandas
def preprocess_weather_data(weather_data):
    # Debugging: Check the structure of the raw weather data
    
    daily_data = weather_data.get('days', [])
    if not daily_data:
        raise HTTPException(status_code=404, detail="No weather data available")

    df = pd.DataFrame(daily_data)

    # Ensure we only include necessary columns and filter out rows with missing values
    df = df[['tempmax', 'tempmin', 'precip', 'windspeed', 'humidity']].dropna()


    # Create a target variable with multiple risk categories
    def risk_category(precip):
        if precip > 100:
            return "High"
        elif precip > 50:
            return "Moderate"
        else:
            return "Low"
    
    df['flood_risk'] = df['precip'].apply(risk_category)
    df['flood_risk'] = pd.Categorical(df['flood_risk']).codes  # Convert to numeric codes for modeling

    # Extract features and target
    features = df[['tempmax', 'tempmin', 'precip', 'windspeed', 'humidity']]
    target = df['flood_risk']

    # Debugging output
    print(f"Features Shape: {features.shape}")
    print(f"Target Shape: {target.shape}")

    # Check lengths before returning
    if features.shape[0] != target.shape[0]:
        raise ValueError("Features and target have different lengths")

    return features, target


# Model Training with Scikit-learn
def train_model(features, target):
    # Standardizing the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)
    
    # Use a RandomForestClassifier for prediction
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Cross-validation for better accuracy assessment
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # Test the model
    predictions = model.predict(X_test)
    
    return model, predictions, cv_scores.mean()

# Endpoint to train the model and predict vulnerability
@app.get("/predict_vulnerability/{city}")
def predict_vulnerability(city: str):
    # Fetch and preprocess the data
    weather_data = get_weather_data(city)
    features, target = preprocess_weather_data(weather_data)
    
    # Train the model
    model, predictions, accuracy = train_model(features, target)

    # Map numerical predictions back to categories
    risk_labels = {0: "Low", 1: "Moderate", 2: "High"}
    predictions_labels = [risk_labels[pred] for pred in predictions]

    return {
        "city": city,
        "predictions": predictions_labels,  # Flood risk predictions for the next days
        "cross_validation_accuracy": accuracy,
    }

@app.get("/current-location/")
def get_current_location():
    url = "https://ipinfo.io"
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    latitude = response.json().get('loc', '').split(',')[0]
    longitude = response.json().get('loc', '').split(',')[1]
    return {"latitude": latitude, "longitude": longitude}

@app.get("/current-weather/")
def get_current_weather(city: str):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/forecast?key={API_KEY}&location={city}&aggregateHours=1&shortColumnNames=true"
    
    response = requests.get(url)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    csv_data = response.text
    
    csv_reader = csv.DictReader(StringIO(csv_data))

    current_conditions = next(csv_reader, None)
    
    if current_conditions is None:
        raise HTTPException(status_code=404, detail="No current weather data found")
    result = {
        "city": current_conditions['name'],
        "temp": current_conditions['temp'],
        "humidity": current_conditions['humidity'],
        "windspeed": current_conditions['wspd'],
        "precip": current_conditions.get('precip', 0),
        "conditions": current_conditions['conditions'],
        "wind_direction": current_conditions['wdir'],
        "cloud_cover": current_conditions['cloudcover'],
    }
    
    return result

@app.get("/weather/")
def get_weather(city: str):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/forecast?key={API_KEY}&location={city}&aggregateHours=24&shortColumnNames=true"
    
    response = requests.get(url)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    csv_data = response.text
    csv_reader = csv.DictReader(StringIO(csv_data))
    
    weather_data = [row for row in csv_reader]
    return weather_data 

def extract_location(description):
    """Extract location coordinates from the description text."""
    try:
        place = description.split(':')[0]  # Customize based on actual feed data
        location = geolocator.geocode(place)
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        print(f"[{current_time()}] Error extracting location: {e}")
    return None


# Background task for continuously checking disaster alerts
def run_disaster_alerts():
    while True:
        check_earthquakes()
        check_tsunamis()
        time.sleep(300)  # Check every 5 minutes


@app.on_event("startup")
def startup_event():
    """Start background tasks on startup."""
    background_tasks.add_task(run_disaster_alerts)


@app.get("/check_disasters/")
def check_disasters():
    """Endpoint for manual checking of earthquakes and tsunamis."""
    check_earthquakes()
    check_tsunamis()
    return {"status": "Alerts checked successfully"}

@app.get("/safe-zones/")
def get_safe_zones(lat: float, lon: float):
    """API endpoint to return safe zones (e.g., hospitals) near a location."""
    try:
        # Replace the query with the proper search criteria
        url = f"https://nominatim.openstreetmap.org/search.php?q=hospital+near+{lat},{lon}&format=jsonv2"
        headers = {
            "User-Agent": "DisasterPredictionApp/1.0 (your_email@example.com)"
        }
        response = requests.get(url, headers=headers)
        
        if response.status_code == 403:
            raise HTTPException(status_code=403, detail="Access blocked: You have violated the usage policy of OSM's Nominatim service. Set a proper User-Agent or reduce request frequency.")
        elif response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching safe zones: {e}")
