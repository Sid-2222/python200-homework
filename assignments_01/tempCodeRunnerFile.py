import pandas as pd
import numpy as np
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    return pd.Series(arr, name= "values")

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    return {
        "mean" : series.mean(),
        "median": series.median(),
        "std" : series.std(),
        "mode" : series.mode()[0]
    }
def data_pipeline(arr):
    series = create_series(arr)
    clear_series = clean_data(series)
    final_summary = summarize_data(clear_series)
    return final_summary

pipe = data_pipeline(arr)

for key, value in pipe.items():
    print(f"{key}: {value}")