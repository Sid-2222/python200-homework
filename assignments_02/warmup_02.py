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

# scikit-learn Question 2

x = np.array([10, 20, 30, 40, 50])
print("Shape:", x.shape)
x = x.reshape(-1,1)
print("Shape:", x.shape)

#  scikit-learn needs X to be 2D because it represent a table of data like row , column 


# scikit-learn Question 3

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3 ,random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)
print("cluster centers")
print(kmeans.cluster_centers_)
print(" points fell into each cluster ")
print(np.bincount(labels))

plt.figure(figsize=(8, 6))
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis', s=60, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color="black", marker="X", s=200)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1 X-axis")
plt.ylabel("Feature 2 Y-axis")
plt.savefig("outputs/kmeans_clusters.png")
plt.show()


# Linear Regression

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Linear Regression Question 1)

plt.scatter(age , cost, c=smoker, cmap="coolwarm",s=40, alpha=0.7)
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")
plt.savefig("outputs/cost_vs_age.png")
plt.show()

# based on the scatter plot there are 2 distinct groups visible smoker and non smoker.
# And it suggest the smoker variable increase the health cost significantly across all ages

# Linear Regression Question 2)
X = age.reshape(-1, 1)
y = cost

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Linear Regression Question 3
model = LinearRegression()
model.fit(X_train,y_train)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)
print("RMSE:", rmse)
print("R2 on the test set:", r2)

# the slop is the increase of the medical bill of each year of increased age.
# whatever is the bill this year next years medicall bill will more as the value of the slope

# Linear Regression Question 4
X_full = np.column_stack([age, smoker])
X_train, X_test, y_train, y_test = train_test_split( X_full, y, test_size=0.2, random_state=42)
model_full = LinearRegression()
model_full.fit(X_train,y_train)
r2 = model_full.score(X_test, y_test)
print("R2 with smoker on the test set:", r2)
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# Adding the smoker flag helped a lot as we can see the R2 value went up to 0.773723 from 0.069515
# The age coefficient (205.17) means every year the amount of health cose increase is $ 205.
# The smoker coefficient (14538.03) means 2 people of the same age the person who smokes will have $ 14.538 more health cost per year than non-smoker

y_pred_full = model_full.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter( y_pred_full, y_test, s=60, alpha=0.7)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--",
    linewidth=2
)
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.savefig("outputs/predicted_vs_actual.png")
plt.show()

# when a point falls above the diagonal line it means th model predicted the value is less than the originalvalue
# if a point is below the line then the model predicted the value more than the actual price