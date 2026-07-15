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

df = pd.read_csv("student_performance_math.csv", sep=";")

print("Shape of dataset: (Rows, Columns) ", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)

plt.hist(df["G3"], bins=21, color="green", rwidth=0.8, edgecolor="black")
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade")
plt.ylabel("Number of students")
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

df_filtered["schoolsup"] = df_filtered["schoolsup"].map({"yes": 1, "no": 0}).astype(int)
df_filtered["internet"] = df_filtered["internet"].map({"yes": 1, "no": 0}).astype(int)
df_filtered["higher"] = df_filtered["higher"].map({"yes": 1, "no": 0}).astype(int)
df_filtered["activities"] = df_filtered["activities"].map({"yes": 1, "no": 0}).astype(int)
df_filtered["sex"] = df_filtered["sex"].map({"M": 1, "F": 0}).astype(int)

original_corr = df["absences"].corr(df["G3"], method="pearson")
filtered_corr = df_filtered["absences"].corr(df_filtered["G3"], method="pearson")
#print(df_filtered.head(10))

print(f"Original: {original_corr}")
print(f"Filtered: {filtered_corr}")

# The original correlation between absences and G3 is very weak because the dataset
# includes students with G3 = 0. These students typically did not take the final exam,
# so their final grade is zero regardless of how many absences they had. Those records
# weaken the relationship between absences and final grades. After removing the G3 = 0
# students, the correlation becomes more negative, showing that among students who
# completed the course, having more absences is associated with lower final grades.

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

# G2 has the strongest positive correlation with G3 (0.965583), which is expected
# because the second-period grade is closely related to the final grade.
# The strongest negative correlation is failures (-0.293831), showing that students
# with more past failures generally earn lower final grades.
# One surprising result is that absences have only a moderate negative correlation
# (-0.213129), suggesting that attendance alone is not as strong a predictor of
# final grades as previous academic performance.

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

# The scatter plot shows a strong positive linear relationship between G2 and G3.
# Students who earn higher second-period grades generally earn higher final grades,
# with relatively little scatter around the overall trend.

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


# The box plot shows that students with no previous failures generally have the
# highest final grades. As the number of past failures increases, the median G3
# decreases. There are a few high-performing outliers among students with multiple
# failures, but the overall trend suggests that previous failures are associated
# with lower final grades.


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

# The slope of -1.4275 means that for each additional past failure, the model predicts
# a student's final grade (G3) will decrease by about 1.43 points on the 0-20 grading scale.
# The RMSE of 2.96 means the model's predictions are off by about 2.96 grade points on average.
# The R² value of 0.0895 means that failures alone explain only about 9% of the variation
# in final grades, so other factors also influence student performance.

# Task 5: Build the Full Model )

feature_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                "absences", "freetime", "goout", "Walc", "schoolsup",
                "internet", "higher", "activities", "sex"]
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

# The full model performed better than the baseline model. The Test R² increased from 0.0895 to 0.2634,
# and the RMSE decreased from 2.9617 to 2.6639. This shows that using more features helps predict 
# students' final grades more accurately than using only past failures.

print("Feature Coefficients")
for name, coef in zip(feature_cols, full_model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# After looking at the data the surprising value is schoolsup : -2.263 which have a negative value shows a negative relation
# Which shows there is no relationship between a student getting Extra educational support from the school will do well in G3

# Comparing train R² (0.2346) and test R² (0.2634), the values are close.
# This means the model performs similarly on unseen data.

# If deploying this model, I would keep features with stronger effects such as failures,
# internet, studytime, and schoolsup because they provide more useful information.

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

# The full model achieved a Test R² of 0.2634 and an RMSE of 2.6639.
# On a 0–20 grade scale, this means the model's predictions are usually off by about
# 2.66 points. The R² value of 0.2634 means the model explains about 26% of the
# variation in students' final grades, so other factors also affect performance.

# Internet has the largest positive coefficient (+1.037), meaning that, after
# accounting for the other variables in the model, students with internet access
# tend to have higher predicted final grades. The second largest positive
# coefficient is sex (+0.402).

# School support has the largest negative coefficient (-2.263), followed by
# failures (-0.800).

# One surprising result was that school support had a negative coefficient.
# This does not mean school support causes lower grades. Instead, after
# accounting for the other variables in the model, students receiving school
# support tended to have lower predicted grades, possibly because they were
# already struggling academically.

# Neglected Feature: The Power of G1

feature_cols_G1 = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                "absences", "freetime", "goout", "Walc", "schoolsup",
                "internet", "higher", "activities", "sex","G1"]
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

# A high R² does not mean G1 causes G3. It only means G1 is strongly
# positively associated with G3 because students who perform well in the
# first grading period also tend to perform well on the final exam.

# Yes, this model is useful for identifying students who may struggle because
# G1 is a strong predictor of the final grade (G3).

# If educators want to identify at-risk students before G1 is available,
# they would need to rely on other features such as previous failures,
# study time, attendance (absences), parental education, school support,
# and other background information to predict which students may need
# extra help early in the school year.