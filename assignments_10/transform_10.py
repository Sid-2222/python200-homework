# https://www.youtube.com/watch?v=MNaGZt1EEa8

import joblib
import json
import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# SETUP
# ============================================================

load_dotenv()

clf = joblib.load("assignments_10/models/weather_classifier.pkl")

with open(
    "assignments_10/models/weather_classifier_metadata.json",
    encoding="utf-8"
) as f:
    metadata = json.load(f)

FEATURES = metadata["feature_names"]
print("FEATURES:", FEATURES)

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ============================================================
# STEP 1: INCREMENTAL READ
# ============================================================

response = (supabase.table("weather_raw").select("*").execute())
raw_rows = response.data

enriched_response = (supabase.table("weather_enriched").select("date").execute())

already_done = {
    row["date"]
    for row in enriched_response.data
}

to_classify = [
    row for row in raw_rows
    if row["date"] not in already_done
]

print("\n--- Incremental Read ---")
print(f"Raw records: {len(raw_rows)}")
print(f"Already enriched: {len(already_done)}")
print(f"Records to process: {len(to_classify)}")


# If everything has already been processed, there is nothing
# else to send through the ML and LLM steps.
if not to_classify:
    print("Nothing to process. All records are already enriched.")
    exit()


# ============================================================
# STEP 2: ML TRANSFORM
# ============================================================

df = pd.DataFrame(to_classify)

print("\nFEATURES:", FEATURES)
print("DF COLUMNS:", df.columns.tolist())
print("DF SHAPE:", df.shape)
print("DF HEAD:")
print(df.head())

# Select only the features expected by the model,
# in exactly the same order used during training.
X = df[FEATURES]

predictions = clf.predict(X)

probabilities = clf.predict_proba(X)[:, 1]

print("\n--- ML Predictions ---")
print(f"Good days predicted: {predictions.sum()} / {len(predictions)}")
print(f"Confidence range: {probabilities.min():.2f} – {probabilities.max():.2f}")


# Build the records that will eventually be written
# to weather_enriched.
enrichment_records = []

for i, row in enumerate(to_classify):
    enrichment_records.append({
        "date": row["date"],
        "good_for_running": bool(predictions[i]),
        "confidence": round(float(probabilities[i]), 4),
        "llm_summary": None
    })


print("\nSample enrichment records:")

for r in enrichment_records[:3]:
    print(r)


good_days = [
    r for r in enrichment_records
    if r["good_for_running"]
]

skip_days = [
    r for r in enrichment_records
    if not r["good_for_running"]
]

print(
    f"\nGood days: "
    f"{len(good_days)} "
    f"({len(good_days) / len(enrichment_records):.0%})"
)

print(f"Skip days: {len(skip_days)}")


# ============================================================
# STEP 3: LLM TRANSFORM
# ============================================================

SYSTEM_PROMPT = (
    "You are writing a one-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write exactly one sentence — direct, practical, and specific to the conditions. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)


def make_user_message(row, good_for_running, confidence):

    prediction_text = (
        "good for running" if good_for_running else "not ideal for running"
    )

    return (
        f"Date: {row['date']}\n"
        f"High: {row['temperature_2m_max']}°C\n"
        f"Low: {row['temperature_2m_min']}°C\n"
        f"Precipitation: {row['precipitation_sum']} mm\n"
        f"Max wind speed: {row['wind_speed_10m_max']} km/h\n"
        f"Model prediction: {prediction_text}\n"
        f"Model confidence: {confidence:.0%}"
    )


def validate_summary(text):
    text = text.strip()
    if not text:
        return None
    
    sentences = [s for s in text.split(".") if s.strip()]
    if len(sentences) > 2:
        return None
    return text

print("\n--- LLM Enrichment Started. Please be patient.---")

for i, record in enumerate(enrichment_records):
    raw_row = next(r for r in to_classify if r["date"] == record["date"])

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_user_message(
                    raw_row, record["good_for_running"], record["confidence"]
                )},
            ],
            max_tokens=100,
        )

        raw_summary = response.choices[0].message.content
        summary = validate_summary(raw_summary) or "Recommendation unavailable."

    except Exception as e:
        print(f"  API error on {record['date']}: {e}")
        summary = "Recommendation unavailable."

    record["llm_summary"] = summary

    if (i + 1) % 50 == 0:
        print(f"LLM enrichment progress: {i + 1} / {len(enrichment_records)}")


# ============================================================
# STEP 4: LOAD
# ============================================================

response = (supabase.table("weather_enriched").upsert(enrichment_records,on_conflict="date").execute())

print("\n--- Load ---")

print( f"Upserted {len(response.data)} rows into weather_enriched")


# ============================================================
# STEP 5: VERIFY
# ============================================================

check = (supabase.table("weather_enriched").select("*").execute())
all_enriched = check.data

print("\n--- Verification ---")

print(f"Total rows in weather_enriched: {len(all_enriched)}")


print("\nFive sample rows:")

for row in all_enriched[:5]:

    print(f"Date: {row['date']} | Good for Running = {row['good_for_running']} | Confidence = {row['confidence']:.2f}")
    print( f"LLM Summary:  {row['llm_summary']}")
    print()


good_count = (supabase.table("weather_enriched").select("date", count="exact").eq("good_for_running", True).execute())

print(f"Good-for-running days in weather_enriched: {good_count.count}")


# The LLM summaries mostly match the weather and the model's predictions.
# The 2023-01-02 summary is a good example because it correctly mentions the comfortable
# temperature, light wind, and low rain. The 2023-11-01 summary seems a little off because
# there was no rain and only and clear sky, but the high temperature may have caused the LLM
# to recommend staying indoors.



# ============================================================
# STEP 6: REFLECTION
# ============================================================

"""
REFLECTION

I used weather data from Ventura, CA. I think the predictions could be somewhat accurate, but maybe not
completely because the model was trained on weather data from Charlotte, NC, which has a different climate.
The LLM recommendation is additive not a replacement for the ML classifier. The classifier determines
the good_for_running label and confidence score, while the LLM turns the weather information and model
output into a human-readable recommendation. This separation means the ML model remains responsible for 
the classification while the LLM provides a more useful explanation for the user.
If the pipeline processed 50,000 records instead of 365, my main concerns would be API cost,and latency. 
It took around 40 seconds to process 50 records so 50,000 will take around 11 hours. A separate LLM API request 
for every record could become expensive. I would handle this by processing the records in smaller groups and making 
sure failed records can be tried again.I would also keep using incremental processing so I do not have to run all 50,000 records again.

"""
