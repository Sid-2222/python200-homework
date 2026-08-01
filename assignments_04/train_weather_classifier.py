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
os.makedirs("outputs", exist_ok=True)


url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude":  34.052235,
    "longitude": -118.243683,
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
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)


print(df.info())
print(df.describe())

#----------------------------------------Step 2: Engineer Labels-----------------------------------------

def label_running_day(row):
    return int(
        7 <= row["temperature_2m_max"] <= 30
        and row["temperature_2m_min"] >= 0
        and row["precipitation_sum"] < 3.0
        and row["wind_speed_10m_max"] < 35
    )

# As i am choosing Los Angels as the city so i changed the max temperature and the max wind speed as its a costal city

df["good_for_running"] = df.apply(label_running_day, axis=1)

print(df.head(10))
print("Class distribution:")
print(df["good_for_running"].value_counts())
print("\nClass distribution (%):")
print(df["good_for_running"].value_counts(normalize=True) * 100)

# The result shows 71% of days are food for running which seems right as per los angles

#----------------------------------------------Step 3: Train and Tune----------------------------------

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[FEATURES]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
])

param_grid = {
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print(f"Best C value: {grid_search.best_params_['clf__C']}")
print(f"Best CV AUC: {grid_search.best_score_:.4f}")

best_pipe = grid_search.best_estimator_
y_pred  = best_pipe.predict(X_test)
y_probs = best_pipe.predict_proba(X_test)[:, 1]

print("\nGridSearchCV classification report")
print(classification_report(y_test, y_pred))
print(f"Test AUC: {roc_auc_score(y_test, y_probs):.4f}")
test_auc = roc_auc_score(y_test, y_probs)
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
fig, ax = plt.subplots(figsize=(6, 5))

RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, name="Logistic Regression")

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random classifier"
)

ax.set_title("ROC Curve — Weather Classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/weather_roc.png")
plt.show()
plt.close()

#--------------------------------Step 4: Reflect on Evaluation----------------------------------

# The AUC score of the model (0.9945) which means it did a great job seperating good running days
# The result is very near 1 and its surprisingly good
# After looking at the classification report we ccan see the recall of class 0 (0.86) is lower than
# class 1 (0.98 ) that means the model will more likely to make false positive errors.
# The model over-recommend running because telling someone not to run on a good day is less useful than
# occasionally suggesting a run on a less ideal day.
# In real life application i would consider lowering the threshold The model over-recommend running because 
# telling someone not to run on a good day is less useful than occasionally suggesting a run on a less ideal day.


#--------------------------------------Step 5: Save the Model--------------------------------------

joblib.dump(best_pipe, "models/weather_classifier.pkl")

metadata = {
    "python_version": sys.version,
    "scikit_learn_version": sklearn.__version__,
    "feature_names": list(X.columns),
    "best_hyperparameters": grid_search.best_params_,
    "test_auc": float(test_auc),
    "location": {
        "city": "Los Angeles",
        "latitude":  34.052235,
        "longitude": -118.243683
    },
    "label_description": (
        "good_for_running = 1 when temperature, precipitation, and wind conditions "
        "met the selected running criteria. Otherwise, the label was set to 0."
    )
}

with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Model and metadata saved successfully.")