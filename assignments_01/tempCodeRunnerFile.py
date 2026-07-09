import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as sat
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns

 # --- Pandas ---
 
# Pandas Q1)
 
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)
print(df.head(3))
print("Shape:", df.shape)
print("Data Type:", df.dtypes)

# Pandas Q2)
grade_above_80 = df[ (df["passed"] == True ) &  (df["grade"] > 80)]
print(grade_above_80)

# Pandas Q3)

df["grade_curved"] = df["grade"] + 5
print(df)

# Pandas Q4)

df["name_upper"] = df["name"].str.upper()
print(df[["name", "name_upper"]])

# Pandas Q5)
by_city = df.groupby(["city"])["grade"].mean()
print(by_city)

# Pandas Q6)
df["city"] = df["city"].replace("Austin", "Houston")
print(df[["name", "city"]])