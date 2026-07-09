from prefect import task, flow
import pandas as pd
import numpy as np

# Pipeline Question 2
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task
def create_series(arr):
    return pd.Series(arr, name= "values")

@task
def clean_data(series):
    return series.dropna()

@task
def summarize_data(series):
    return {
        "mean" : series.mean(),
        "median": series.median(),
        "std" : series.std(),
        "mode" : series.mode()[0]
    }
 
@flow    
def pipeline_flow(arr):
    series = create_series(arr)
    clear_series = clean_data(series)
    final_summary = summarize_data(clear_series)
    return final_summary

if __name__ == "__main__":
    result = pipeline_flow(arr)
    for key , value in result.items():
        print(f"{key}: {value}")
        
# Q1. Yes here its a simple example to introduce prefect but there are more powerful thing prefect can do so the introduction is needed.
# Q2. A real life case can be getting daily sales report and schedule it to run every morning