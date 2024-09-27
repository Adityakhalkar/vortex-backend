import requests
from geopy.distance import geodesic
import time
import smtplib
from email.mime.text import MIMEText
import feedparser
from geopy.geocoders import Nominatim
from datetime import datetime

# Configuration
USGS_EARTHQUAKE_API = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson"
TSUNAMI_RSS_FEED = "https://www.tsunami.gov/rss/tsunami_alerts.xml"  # Replace with the actual RSS feed URL
CHECK_INTERVAL = 300  # in seconds (5 minutes)
PROXIMITY_KM = 100  # Alert if event is within 100 km
EMAIL_ALERT = True  # Set to True to enable email alerts

# Email Configuration (if EMAIL_ALERT is True)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = "your_email@gmail.com"
EMAIL_PASSWORD = "your_password"
RECIPIENT_EMAIL = "recipient_email@gmail.com"

# Your location
USER_LOCATION = (18.5196, 73.8554)  # Example: San Francisco

# To keep track of alerted events to avoid duplicate alerts
alerted_earthquakes = set()
alerted_tsunamis = set()

# Initialize geolocator
geolocator = Nominatim(user_agent="tsunami_alert_system")

def send_email(subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECIPIENT_EMAIL

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        print(f"[{current_time()}] Email sent successfully: {subject}")
    except Exception as e:
        print(f"[{current_time()}] Failed to send email: {e}")

def current_time():
    """Returns the current time formatted as a string."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def check_earthquakes():
    alerts_found = False
    try:
        response = requests.get(USGS_EARTHQUAKE_API)
        data = response.json()
        for feature in data['features']:
            quake_id = feature['id']
            if quake_id in alerted_earthquakes:
                continue  # Skip already alerted events
            coords = feature['geometry']['coordinates']
            event_location = (coords[1], coords[0])  # (latitude, longitude)
            distance = geodesic(USER_LOCATION, event_location).kilometers
            magnitude = feature['properties']['mag']
            if distance <= PROXIMITY_KM:
                alerts_found = True
                alert_message = (
                    f"🌎 **Earthquake Alert!** 🌎\n\n"
                    f"**Magnitude:** {magnitude}\n"
                    f"**Location:** {feature['properties']['place']}\n"
                    f"**Distance:** {distance:.2f} km\n"
                    f"**Time:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(feature['properties']['time']/1000))}"
                )
                print(f"[{current_time()}] {alert_message}")
                if EMAIL_ALERT:
                    send_email("Earthquake Alert", alert_message)
                alerted_earthquakes.add(quake_id)
    except Exception as e:
        print(f"[{current_time()}] Error checking earthquakes: {e}")
    
    if not alerts_found:
        print(f"[{current_time()}] ✅ No earthquake alerts. Everything is fine.")

def check_tsunamis():
    alerts_found = False
    try:
        feed = feedparser.parse(TSUNAMI_RSS_FEED)
        for entry in feed.entries:
            tsunami_id = entry.id if 'id' in entry else entry.link
            if tsunami_id in alerted_tsunamis:
                continue  # Skip already alerted tsunamis
            # Parse necessary information from the entry
            title = entry.title
            description = entry.description
            published = entry.published

            # Extract location information from the description
            tsunami_location_coords = extract_location(description)
            if not tsunami_location_coords:
                continue  # Unable to parse location

            distance = geodesic(USER_LOCATION, tsunami_location_coords).kilometers
            if distance <= PROXIMITY_KM:
                alerts_found = True
                alert_message = (
                    f"🌊 **Tsunami Alert!** 🌊\n\n"
                    f"**Title:** {title}\n"
                    f"**Description:** {description}\n"
                    f"**Distance:** {distance:.2f} km\n"
                    f"**Time:** {published}"
                )
                print(f"[{current_time()}] {alert_message}")
                if EMAIL_ALERT:
                    send_email("Tsunami Alert", alert_message)
                alerted_tsunamis.add(tsunami_id)
    except Exception as e:
        print(f"[{current_time()}] Error checking tsunamis: {e}")
    
    if not alerts_found:
        print(f"[{current_time()}] ✅ No tsunami alerts. Everything is fine.")

def extract_location(description):
    """
    Extract location coordinates from the description text.
    This function assumes that the description contains location names.
    You may need to customize this based on the actual RSS feed content.
    """
    try:
        place = description.split(':')[0] 
        location = geolocator.geocode(place)
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        print(f"[{current_time()}] Error extracting location: {e}")
    return None

