from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd



# --- scikit-learn API --- #

##  scikit-learn Question 1

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])
new_value = np.array([4,8]).reshape(-1,1)
model = LinearRegression()
model.fit(years,salary)
y_predicted = model.predict(new_value)

print(f"Slope: {model.coef_[0]:,.5f}")
print(f"Intercept: {model.intercept_:,.5f}")
print(f"Predicted salary for 4 years of experience = ${y_predicted[0]:,.2f}")
print(f"Predicted salary for 8 years of experience = ${y_predicted[1]:,.2f}")
