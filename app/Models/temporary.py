import requests
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# URL of your backend API
API_URL = "https://localhost:7025/api/ml/logs"

# Skip SSL verification for localhost HTTPS
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Fetch prediction logs
response = requests.get(API_URL, verify=False)
logs = response.json()

# Extract humidity logs
timestamps = []
humidity_values = []

for log in logs:
    if log['sensorType'].lower() == "humidity":
        timestamps.append(datetime.fromisoformat(log['sensorTimestamp'].replace('Z', '+00:00')))
        humidity_values.append(log['value'])

# Convert timestamps to numbers for regression
time_numbers = np.array([(ts - timestamps[0]).total_seconds() / 3600 for ts in timestamps])  # hours since first reading
humidity_values = np.array(humidity_values)

# Fit a linear regression line
if len(time_numbers) > 1:
    coeffs = np.polyfit(time_numbers, humidity_values, 1)  # degree 1 polynomial (line)
    trend_line = np.poly1d(coeffs)
    trend_values = trend_line(time_numbers)

# Plot humidity readings
plt.figure(figsize=(12, 6))
plt.scatter(timestamps, humidity_values, color='blue', label='Humidity Readings')

# Plot trend line
if len(time_numbers) > 1:
    plt.plot(timestamps, trend_values, color='red', linestyle='--', label='Trend Line')

plt.axhline(y=30, color='gray', linestyle='--', label='Watering Threshold (30%)')
plt.xlabel('Timestamp')
plt.ylabel('Humidity (%)')
plt.title('Soil Humidity Over Time with Trend Line')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
