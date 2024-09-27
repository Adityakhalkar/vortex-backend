from fastapi import FastAPI
import requests
from dotenv import load_dotenv
from gdacs.api import GDACSAPIReader
import os

load_dotenv()

app = FastAPI()

API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")

@app.get("/")
def read_root():
    return "Vortexa API"

@app.get("/weather/{city}")
def get_weather(city: str):
    if not API_KEY:
        return {"error": "API key not found"}

    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}"
    params = {
        "unitGroup": "metric",
        "key": API_KEY,
        "include": "current"
    }
    
    response = requests.get(url, params=params)

    if response.status_code != 200:
        return {"error": f"Failed to fetch data for {city}"}
    
    return response.json()