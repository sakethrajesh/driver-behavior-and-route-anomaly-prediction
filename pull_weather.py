import concurrent.futures
import csv  # Import the csv module for incremental writing
import warnings
from datetime import datetime

import pandas as pd
from meteostat import Hourly, Point
from tqdm import tqdm

# Load the dataset
data = pd.read_csv('dataset/EVChargingStationUsage.csv')

# Define the output file
output_file = 'dataset/EVChargingWithWeather.csv'

# Write header row to the file
with open(output_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    # Combine original headers with new weather data headers
    writer.writerow(list(data.columns) + ['temp', 'rhum', 'prcp'])

# Function to truncate datetime to the hour
warnings.simplefilter(action='ignore', category=FutureWarning)


def truncate_to_hour(date_time_str):
    date_time_str = date_time_str.split(":")[0]  # Keep only "7/29/2011 20"
    start_time = datetime.strptime(date_time_str, "%m/%d/%Y %H")
    return start_time


# Function to fetch weather data for a given location and time
def fetch_weather(lat, lon, date_time):
    date_time_hourly = truncate_to_hour(date_time)
    location = Point(lat, lon)
    weather = Hourly(location, date_time_hourly, date_time_hourly).fetch()
    if not weather.empty:
        return weather.iloc[0].to_dict()
    else:
        return {"temp": None, "rhum": None, "prcp": None}


# Function to process each row, fetch weather data, and write to file
def process_row(row):
    weather = fetch_weather(
        row["Latitude"], row["Longitude"], row["Start Date"])
    result = list(row) + [weather.get("temp"),
                          weather.get("rhum"), weather.get("prcp")]

    # Write the row to the file
    with open(output_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)


# Use ThreadPoolExecutor to process rows concurrently
with concurrent.futures.ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(process_row, [row for _, row in data.iterrows()]),
              total=len(data),
              desc="Processing rows with weather data",
              ncols=100))
