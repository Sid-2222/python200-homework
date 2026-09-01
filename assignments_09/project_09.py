import requests
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 34.28,
    "longitude": -119.29,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Los_Angeles",
}

response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()

print("\nAPI response received successfully")
print("\nResponse keys:", data.keys())
print("\nNumber of daily records:", len(data["daily"]["time"]))

daily = data["daily"]

records = [
    {
        "date": daily["time"][i],
        "temperature_2m_max": daily["temperature_2m_max"][i],
        "temperature_2m_min": daily["temperature_2m_min"][i],
        "precipitation_sum": daily["precipitation_sum"][i],
        "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
    }
    for i in range(len(daily["time"]))
]


print("\nFirst record:", records[0])
print("\nLast record:", records[-1])

# A full year in 2023 has 365 days because 2023 was not a leap year. I expect 365 records and i got 365 days records. 
# If I get a different number, some dates may be missing from the API response or the date range may not cover the full year.

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
response = (supabase.table("weather_raw").upsert(records, on_conflict="date").execute())
print(f"\nUpserted {len(response.data)} rows into weather_raw")



# Running it multiple times it doesn't duplicates or add more rows to the table
# This shows that the pipeline is idempotent: running it multiple times
# produces the same final result instead of adding duplicate records.


count_response = (supabase.table("weather_raw").select("date", count="exact").execute())
print(f"\nRows in weather_raw: {count_response.count}")

earliest = (supabase.table("weather_raw").select("date").order("date", desc=False).limit(1).execute())
print(f"\nEarliest date: {earliest.data[0]['date']}")

latest = (supabase.table("weather_raw").select("date").order("date", desc=True).limit(1).execute())
print(f"\nLatest date: {latest.data[0]['date']}")

july_4 = (supabase.table("weather_raw").select("*").eq("date", "2023-07-04").execute())
print(f"\n2023-07-04: {july_4.data[0]}")