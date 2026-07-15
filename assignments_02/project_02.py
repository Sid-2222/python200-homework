# From watching the raw dataset i noticed fields are separated by ";"
# so we will use sep=";" when reading it with pd.read_csv().


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

# Task 1: Load and Explore

df = pd.read_csv("outputs/student_performance_math.csv", sep=";")

print("Shape of dataset: (Rows, Columns) ", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)

plt.hist(df["G3"], bins=21, color="green", rwidth=0.8, edgecolor="black")
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final grade G3")
plt.ylabel("Number of student")
plt.savefig("outputs/g3_distribution.png")
plt.show()

# Task 2: Preprocess the Data

df_filtered = df[df["G3"] > 0].copy()
print("Shape of filtered dataset: (Rows, Columns)", df_filtered.shape)
print("Rows removed:", df.shape[0] - df_filtered.shape[0])

# we are removing 0 values from G3 because they dont represent real scores 
# they represents students who didnt took their G3 exams.
# So keeping those value in calculation will distort the model


# converting the yes/no columns to 1/0 and the sex column to 0/1

df_filtered.loc[df_filtered["schoolsup"] == "yes", "schoolsup"] = 1
df_filtered.loc[df_filtered["schoolsup"] == "no", "schoolsup"] = 0

df_filtered.loc[df_filtered["internet"] == "yes", "internet"] = 1
df_filtered.loc[df_filtered["internet"] == "no", "internet"] = 0

df_filtered.loc[df_filtered["higher"] == "yes", "higher"] = 1
df_filtered.loc[df_filtered["higher"] == "no", "higher"] = 0

df_filtered.loc[df_filtered["activities"] == "yes", "activities"] = 1
df_filtered.loc[df_filtered["activities"] == "no", "activities"] = 0

df_filtered.loc[df_filtered["sex"] == "M", "sex"] = 1
df_filtered.loc[df_filtered["sex"] == "F", "sex"] = 0

original_corr = df["absences"].corr(df["G3"], method="pearson")
filtered_corr = df_filtered["absences"].corr(df_filtered["G3"], method="pearson")
#print(df_filtered.head(10))

print(f"Original: {original_corr}")
print(f"Filtered: {filtered_corr}")

# Filtering changes the correlation because students with G3=0 are likely failing students.
# In the original dataset, these students often had very low final grades regardless of their
# number of absences, adding many points that weaken the relationship between absences and G3.
# After removing G3=0 cases, the remaining students show a clearer pattern where higher
# absences are associated with slightly lower final grades.

# Original data
plt.scatter(df["absences"], df["G3"])
plt.xlabel("Absences")
plt.ylabel("G3")
plt.title("Absences vs G3 (Original Dataset)")
plt.show()

# Filtered data
plt.scatter(df_filtered["absences"], df_filtered["G3"])
plt.xlabel("Absences")
plt.ylabel("G3")
plt.title("Absences vs G3 (Filtered Dataset)")
plt.show()

#Task 3: Exploratory Data Analysis
corr_with_g3 = df_filtered.corr(method="pearson")["G3"].drop("G3").sort_values()
print("Correlation between each numeric feature and G3")
print(corr_with_g3)

# The G2 feature has the strongest relationship with G3 with the value 0.965583
# The surprising result is the relationship between G3 and failures is the lowest value -0.293831

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df_filtered,
    x="G2",
    y="G3"
)

plt.title("Relationship Between Second Period Grade (G2) and Final Grade (G3)")
plt.xlabel("G2 Grade")
plt.ylabel("Final Grade (G3)")
plt.savefig("outputs/g2_vs_g3.png", bbox_inches="tight")
plt.show()
plt.close()

# From the Scatter plot we can see a consistant positive correlation between G2 and G3
# From that we can predict a student who scored high will most likely to score high in G3 too

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df_filtered,
    x="failures",
    y="G3"
)

plt.title("Past Failures and Final Grade Performance")
plt.xlabel("Number of Past Failures")
plt.ylabel("Final Grade (G3)")
plt.savefig("outputs/failures_vs_g3.png", bbox_inches="tight")
plt.show()
plt.close()

# from this box plot we can see students with 0 past failures got the highest number and also with the highest mean 
# students with 1 past failures performed little less than students with 0 past failures
# students with 2 past failures performed little less than students with 1 past failures but there are students
# who performed well because we can see some outliers


# Task 4: Baseline Model )
X = df_filtered[["failures"]]
y = df_filtered["G3"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Task 4 Model")
print("Slope:", model.coef_[0])
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)
print("RMSE:", rmse)
print("R2 on the Filtered Dataset:", r2)

# The slope -1.4275 tells us that the model predict -1.4275 less number with every past failures
# The RSME tells us the model's prediction is off on avarage by 2.961737
# The R² value is 0.08949 which is less than my expectation that means predicting G3 only 
# based on failures isnt enough there are more factors that comes into factors

# Task 5: Build the Full Model )

feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
X = df_filtered[feature_cols].values
y = df_filtered["G3"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

full_model = LinearRegression()
full_model.fit(X_train, y_train)

y_train_pred = full_model.predict(X_train)
y_test_pred = full_model.predict(X_test)

train_r2 = full_model.score(X_train, y_train)
test_r2 = full_model.score(X_test, y_test)
rmse = np.sqrt(np.mean((y_test_pred - y_test) ** 2))

print("\nTask 5 Full Model")
print("Train R²:", train_r2)
print("Test R²:", test_r2)
print("RMSE:", rmse)

# The full model performed better than the baseline model. The Test R² increased from 0.0895 to 0.1539,
# and the RMSE decreased from 2.9617 to 2.8550. This shows that using more features helps predict 
# students' final grades more accurately than using only past failures.

print("Feature Coefficients")
for name, coef in zip(feature_cols, full_model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# After looking at the data the surprising value is schoolsup : -2.062 which have a negative value shows a negative relation
# Which shows there is no relationship between a student getting Extra educational support from the school will do well in G3

# Comparing train R² (0.1749) and test R² (0.1539), the values are close.
# This means the model performs similarly on unseen data.

# If deploying this model, I would keep features with stronger effects such as failures,
# higher, internet, studytime, and schoolsup because they provide more useful information.

# I would consider dropping activities and freetime because their coefficients are close to
# zero and they add little value to the prediction.

# Task 6: Evaluate and Summarize

plt.figure(figsize=(8, 6))
plt.scatter(y_test_pred, y_test, color="blue", alpha=0.7)
plt.plot([0, 20], [0, 20], color="red", linestyle="--")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")
plt.savefig("outputs/predictedG3_vs_actual.png", bbox_inches="tight")
plt.show()
plt.close()


# The filtered dataset contains 357 students, and the test set contains about 72 students
# (20% of the filtered dataset).

# The full model achieved a Test R² of 0.1539 and an RMSE of 2.8550.
# On a 0-20 grade scale, this means the model's predictions are usually off by about
# 2.86 points from the actual value. The R² value means the model correctly worked about 15% in
# students' final grades, so there are still other factors affceting performance.

# Internet has the largest positive effect +0.834, meaning students with internet
# usually get higher final grades. School support has the largest negative
# effect -2.062, which is surprising.

# One surprising result was that school support had a negative effect.
# I expected it to help students, but it may be because students who
# receive extra support are already struggling in school.

# Neglected Feature: The Power of G1

feature_cols_G1 = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime","G1"]
X = df_filtered[feature_cols_G1].values
y = df_filtered["G3"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

full_model = LinearRegression()
full_model.fit(X_train, y_train)

y_train_pred = full_model.predict(X_train)
y_test_pred = full_model.predict(X_test)

train_r2 = full_model.score(X_train, y_train)
test_r2 = full_model.score(X_test, y_test)
rmse = np.sqrt(np.mean((y_test_pred - y_test) ** 2))

print("\nTask 6 Full Model with G1")
print("With G1 Test R²:", test_r2)
print("With G1 RMSE:", rmse)

# A high R² does not mean G1 causes G3. It only means G1 is strongly positivly related
# to G3 because students who do well in the first period often do well in the
# final exam.

#  Yes, this model is useful for identifying students who may struggle because G1
# is a strong predictor of the final grade.

# If the educators want to help students before G1 is available, they would need
# to use other information such as past failures, study time, attendance,
# parental education, and school support to predict which students may need
# extra help earlier so students can perform good in G1 so they can perform good in G3 as well.