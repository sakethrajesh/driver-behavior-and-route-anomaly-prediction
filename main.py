import concurrent.futures
import warnings
from datetime import datetime

import pandas as pd
from meteostat import Hourly, Point
from tqdm import tqdm  # Import tqdm for the progress bar

# Load the dataset
data = pd.read_csv('dataset/EVChargingStationUsage.csv')

# Function to truncate datetime to the hour
warnings.simplefilter(action='ignore', category=FutureWarning)


def truncate_to_hour(date_time_str):
    date_time_str = date_time_str.split(":")[0]  # Keep only "7/29/2011 20"
    # Convert the given date and time to a datetime object (ignore minutes)
    start_time = datetime.strptime(date_time_str, "%m/%d/%Y %H")
    return start_time

# Function to fetch weather data for a given location and time


def fetch_weather(lat, lon, date_time):
    date_time_hourly = truncate_to_hour(date_time)
    location = Point(lat, lon)
    weather = Hourly(location, date_time_hourly, date_time_hourly).fetch()
    if not weather.empty:
        # Convert the first row of data to a dictionary
        return weather.iloc[0].to_dict()
    else:
        # Return None for missing data
        return {"temp": None, "rhum": None, "prcp": None}

# Function to fetch weather data in parallel


def fetch_weather_concurrently(row):
    return fetch_weather(row["Latitude"], row["Longitude"], row["Start Date"])


# Use ThreadPoolExecutor to fetch weather data in parallel with a progress bar
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Wrap the executor.map with tqdm for a progress bar
    weather_data = list(tqdm(executor.map(fetch_weather_concurrently, [row for _, row in data.iterrows()]),
                             total=len(data),
                             desc="Fetching weather data",
                             ncols=100))

# Create a DataFrame from the weather data
weather_df = pd.json_normalize(weather_data)

# Add weather data to the original DataFrame
df = pd.concat([data, weather_df], axis=1)

# Save the updated DataFrame with weather data to a CSV file
df.to_csv('dataset/EVChargingWithWeather.csv', index=False)

# Print the head of the DataFrame to verify
print(df.head())
