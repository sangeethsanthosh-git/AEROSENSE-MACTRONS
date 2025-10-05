import pandas as pd

# Load your existing dataset
city_day = pd.read_csv(r'city_air\city_day.csv')

# Load the WAQI dataset, skipping comment lines
waqi_data = pd.read_csv(r'city_air\waqi-covid19-airqualitydata-2025.csv', comment='#')

# Filter WAQI data for Indian cities and years 2021–2025
indian_cities = city_day['City'].unique()
waqi_data = waqi_data[waqi_data['city'].isin(indian_cities)]
waqi_data = waqi_data[waqi_data['date'].str.contains('202[1-5]')]

# Rename columns to match city_day.csv
waqi_data = waqi_data.rename(columns={
    'city': 'City',
    'date': 'Date',
    'pm25': 'PM2.5',
    'pm10': 'PM10',
    'no2': 'NO2',
    'o3': 'O3',
    'co': 'CO',
    'so2': 'SO2',
    'aqi': 'AQI'
})

# Add AQI_Bucket based on AQI values (using Indian AQI standard)
def assign_aqi_bucket(aqi):
    try:
        aqi = float(aqi)
        if aqi <= 50:
            return 'Good'
        elif aqi <= 100:
            return 'Satisfactory'
        elif aqi <= 200:
            return 'Moderate'
        elif aqi <= 300:
            return 'Poor'
        elif aqi <= 400:
            return 'Very Poor'
        else:
            return 'Severe'
    except (ValueError, TypeError):
        return None

waqi_data['AQI_Bucket'] = waqi_data['AQI'].apply(assign_aqi_bucket)

# Add missing columns with NaN if not present in WAQI data
for col in city_day.columns:
    if col not in waqi_data.columns:
        waqi_data[col] = pd.NA

# Ensure Date is in the correct format
waqi_data['Date'] = pd.to_datetime(waqi_data['Date'])

# Extract Year and Month for consistency
waqi_data['Year'] = waqi_data['Date'].dt.year
waqi_data['Month'] = waqi_data['Date'].dt.month

# Select only the columns present in city_day.csv
waqi_data = waqi_data[city_day.columns]

# Append the new data to city_day.csv
updated_city_day = pd.concat([city_day, waqi_data], ignore_index=True)

# Remove duplicates (if any)
updated_city_day = updated_city_day.drop_duplicates(subset=['City', 'Date'], keep='first')

# Save the updated dataset
updated_city_day.to_csv(r'city_air\city_day_updated.csv', index=False)
print("Updated dataset saved as city_air/city_day_updated.csv")