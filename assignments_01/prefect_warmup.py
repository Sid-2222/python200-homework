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
@task
def show_result(final_summary):
     for key , value in final_summary.items():
        print(f"{key}: {value}")
@flow    
def pipeline_flow():
    series = create_series(arr)
    clear_series = clean_data(series)
    final_summary = summarize_data(clear_series)
    show_result(final_summary)
    return final_summary

if __name__ == "__main__":
    pipeline_flow()
    
        
# Prefect Reflection Questions
#
# Q1. Prefect adds some overhead compared to running normal Python functions because
# it needs to track tasks, manage execution states, store logs, and provide workflow
# features. However, this extra overhead is useful for larger data pipelines because
# it makes workflows easier to monitor, debug, retry, and maintain.
#
# Q2. A realistic use case for Prefect would be an automated daily business reporting
# pipeline. For example, a company could schedule a workflow that collects sales data,
# cleans and validates the data, calculates important metrics, and generates a report
# every morning. Prefect would help track each step and handle failures automatically.