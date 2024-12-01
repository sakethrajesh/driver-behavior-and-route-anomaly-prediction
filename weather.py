from datetime import datetime

from meteostat import Daily, Hourly, Point

# Given data
latitude = 37.444572
longitude = -122.160309
date_time_str = "7/29/2011 20:17"
date_time_str = date_time_str.split(":")[0]  # Keep only "7/29/2011 20"

# Convert the given date and time to a datetime object (ignore minutes)
start_time = datetime.strptime(date_time_str, "%m/%d/%Y %H")
# Meteostat works with hourly data; set the end time to the same hour
end_time = start_time

# Create a Point for the location
location = Point(latitude, longitude)

# Fetch hourly weather data
weather_data = Hourly(location, start_time, end_time)
weather_data = weather_data.fetch()

# Display the weather data
if not weather_data.empty:
    print(weather_data)
else:
    print(f"No weather data available for {
          start_time} at ({latitude}, {longitude})")
