from datetime import date
import os
from dotenv import load_dotenv
from supabase import create_client
import supabase
##------------------------------------------Part 1: Warmup---------------------------------------------##

##-----------------------------------------Supabase Connection-------------------------------------##

##-----------------------------------------Connection Question 1-------------------------------------##


""" 
    The two pieces of information supabase-py needs to connect to project are SUPABASE_URL and SUPABASE_KEY.

    To get the SUPABASE_URL/PROJECT_URL, goto the peoject dashboard then under the project name there will be the
    url and can copy it from the dropdown.

    To get the SUPABASE_KEY , goto the Project Dashboard -> Project Settings -> API Keys -> Legacy anon, service_role API keys
    There the key named anon public and be able to copy it.

    We should never be hard code those keys in a Python script because its a security risk. and is soneone gets it can access the
    database.

"""


##-----------------------------------------Connection Question 2-------------------------------------##


def get_client():
    load_dotenv()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)

get_client()
##-----------------------------------------Connection Question 3-------------------------------------##


""" 
Row Level Security (RLS) is a security feature that controls which row a user can access or modify in a database.

For this course I disabled it because its a learning environment and we want to focus on the ELT process.So we are skipping
the cpmolexity of authentication and authorization.

I want to have it enabled is a real world application where the data are sensetive like medical records or financial data.


"""

##-----------------------------------------supabase-py CRUD-------------------------------------##

##-----------------------------------------CRUD Question 1--------------------------------------##

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
def insert_test_record(supabase):
    row = {
        "date": date.today().isoformat(),
        "temperature_2m_max": 78.0,
        "temperature_2m_min": 58.0,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 12.0,
    }

    return supabase.table("weather_raw").insert(row).execute()

#result = insert_test_record(supabase)
#print(f"Inserted record: {result.data}")

# If I ran this function twice, the second insert would fail becausethe date column is the PRIMARY KEY,
# so today's date can only appear once.
# To make the function safe to run multiple times, I would use upsert()
# instead of insert(). With date as the conflict column.
# supabase.table("weather_raw").upsert(row, on_conflict="date").execute()


##-----------------------------------------CRUD Question 2--------------------------------------##


def get_records_by_date_range(supabase, start, end):
    response = (
        supabase.table("weather_raw")
        .select("*")
        .gte("date", start)
        .lte("date", end)
        .execute()
    )

    return response.data

result = get_records_by_date_range(
    supabase,
    "2023-12-31",
    "2026-08-31"
)
print(f"Records from your choosen dates:\n")
print(result)


##-----------------------------------------CRUD Question 3--------------------------------------##


# insert() adds new rows to the table. If a row already exists with the ame primary key, 
# the insert will give an error.
# upsert() adds new rows, but if a row already exists with the same
# conflict key, it updates that existing row instead of giving us an error.

# If I know that every weather record is completely new and I don't want any duplicates, I would use insert() 

# If I am loading weather data repeatedly and want to safely update an existing date or add a new date, I would use upsert().

def safe_upsert(supabase, records):
    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )

    print(f"Rows affected: {len(response.data)}")
    return response.data

records = [
    {
        "date": "2026-08-31",
        "temperature_2m_max": 80.0,
        "temperature_2m_min": 60.0,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 10.0
    },
    {
        "date": "2026-09-01",
        "temperature_2m_max": 82.0,
        "temperature_2m_min": 61.0,
        "precipitation_sum": 0.1,
        "wind_speed_10m_max": 14.0
    }
]

safe_upsert(supabase, records)


##-----------------------------------------Idempotency--------------------------------------##

##---------------------------------------Idempotency Question 1--------------------------------------##


"""

Idempotency is important in a data pipeline because we may need to run the same script more than once, 
especially if it crashes or needs to be restarted. An idempotent pipeline makes sure that running it again does
not create duplicate or incorrect data.

For example, A script is loading weather records for 30 days. It successfully inserts the first 15 days, but then crashes.
If we restart the script and it tries to insert those same 15 days again,it could create duplicate records or 
cause primary-key errors. Using upsert() makes the pipeline safer because existing records can be updated instead of inserted again.


"""