import requests
import pandas as pd
import joblib
import sklearn
import sys
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import os


loaded_model = joblib.load("models/weather_classifier.pkl")
with open("models/weather_classifier_metadata.json", "r") as f:
    metadata = json.load(f)

print("\nModel Metadata:")
print(f"City: {metadata['location']['city']}")
print(f"Latitude: {metadata['location']['latitude']}")
print(f"Longitude: {metadata['location']['longitude']}")
print("\nFeatures:")
for feature in metadata["feature_names"]:
    print(f"{feature}")
print(f"\nTest AUC: {metadata['test_auc']:.4f}")

new_days = pd.DataFrame({
    "temperature_2m_max": [18.0, 22.0, 13.0, 5.0, 29.0, 15.0],
    "temperature_2m_min":  [10.0, 14.0, 8.0, -2.0, 1.0, 9.0],
    "precipitation_sum":  [0.0,  0.2, 25.0, 0.0, 2.5, 3.0],
    "wind_speed_10m_max": [12.0, 15.0, 40.0, 35.0, 33.0, 22.0],
})

pred_labels = loaded_model.predict(new_days)
pred_probs = loaded_model.predict_proba(new_days)[:, 1]

for i, row in new_days.iterrows():
    print(f"\nDay {i + 1}")
    print("Conditions:")
    print(row)
    label = "good" if pred_labels[i] == 1 else "skip"
    print(f"Predicted label: {label}")
    print(f"Confidence (good for running): {pred_probs[i]:.4f}")
    
# The borderline case I included was Day 6, with a maximum temperature of 15.0°C,
# minimum temperature of 9.0°C, 3.0 mm of precipitation, and wind speed of 22.0.
# The model predicted "good" with a probability of 0.8975, so it was fairly confident
# in recommending this day for running.
# If the model predicted a day with a probability of 0.52, I would consider it a very uncertain prediction 
# Also interesting Day 5 with maximum temperature 29.0, minimum temperature 1.0, 2.5 mm of precipitation, 
# wind speed of  33.0 is labeled as skip

# The training script and prediction script are completely separate, so predict_weather.py
# depends on the saved model file and metadata already existing. If someone ran
# predict_weather.py before train_weather_classifier.py, it would fail because the
# models/weather_classifier.pkl and models/weather_classifier_metadata.json files would
# not be available. To make the error more helpful, I would check if the files exist
# before loading them and print a message explaining that the model needs to be trained
# first.

# In a production system, predict_weather.py would need to get new weather forecast data
# automatically instead of using manually created examples. It would need to call a
# weather API for tomorrow's forecast, convert the returned data into the same feature
# format used during training, load the saved pipeline, and generate the running
# recommendation and confidence score automatically each day.