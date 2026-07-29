import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.multiclass import OneVsRestClassifier

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target


# --- Preprocessing ---

# ----------Preprocessing Question 1 ------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")


# ------------Preprocessing Question 2-----------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(np.mean(X_train_scaled, axis=0))

# I fit the scaler on X_train only to prevent data leak, so it dont have the test data.




# ----------------- KNN -------------------------

# --------------------KNN Question 1------------------

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------KNN Question 2--------------------

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)
print("Accuracy Scaled:", accuracy_score(y_test, y_pred_scaled))

# After comparing the both dataset accuracy its visible the accuracy in unscaled dataset 100%
# and the accuracy in scaled dataset is 93% so scaling hurt the performance by 7%
# Beacuse the dataset features are already in similar scale

# -----------------------KNN Question 3---------------------

cv_scores = cross_val_score(knn, X_train, y_train, cv=5)

print("Fold scores:", cv_scores)
print("Mean accuracy:", np.mean(cv_scores))
print("Standard deviation:", np.std(cv_scores))

# The result is more trust worthy then a single train test because
# here the dataset is is divided into 5 subsets of training data. so it will be giving more stable outputs



#----------------------------------KNN Question 4---------------------------------

k_values = [1, 3, 5, 7, 9, 11, 13, 15]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k value = {k}, Mean CV Accuracy = {np.mean(cv_scores):.5f}")
    
# I would chooce k = 5 or k = 7 because it gives us the most accurate result 97.5 %


# -----------------------------Classifier Evaluation------------------------------

# -------------------------Classifier Evaluation Question 1-----------------------

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap="Blues")
plt.savefig("outputs/knn_confusion_matrix.png", bbox_inches="tight")
#plt.show()
plt.close()

# The confusion matrix does not confuse in any species and predicts with 100% accuracy


#--------------------------The sklearn API: Decision Trees-----------------------------

#--------------------------Decision Trees Question 1---------------------------

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train,y_train)
y_pred_dt= dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

# The Decision Tree Accuracy was 96 %  where the knn was 100% accurate so the knn performed 4% better
# Scaling don't significantly affect Decision Tree performance because decision trees do not use distance calculations



#------------------------Logistic Regression and Regularization----------------------------

#-------------------Logistic Regression Question 1---------------------------------------

    
for C in [0.01, 1.0, 100]:
    log_reg = OneVsRestClassifier(
        LogisticRegression(
            C=C,
            max_iter=1000,
            solver="liblinear",
        )
    )
    log_reg.fit(X_train_scaled, y_train)
    coef_sum = sum(np.abs(est.coef_).sum() for est in log_reg.estimators_)
    print(f"C={C}, total coefficient magnitude={coef_sum}")
    
# when the total coefficient magnitude (C) increases the coefficient magnitude also increases
# The larger the C is the weaker is the regulation smaller C is stronger regulations.



#----------------------------------PCA-----------------------------------------

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

#-------------------------------------PCA Question 1-----------------------------

print("Shape of X_digits:", X_digits.shape)
print("Shape of images:", images.shape)
fig, axes = plt.subplots(1, 10, figsize=(10, 3))
for digit in range(10):
    index = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[index], cmap="gray_r")
    axes[digit].set_title(str(digit))
    axes[digit].axis("off")
plt.tight_layout()
plt.savefig("outputs/sample_digits.png", bbox_inches="tight")
#plt.show()
plt.close()

#----------------------------PCA Question 2-----------------------------

pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10) 
plt.colorbar(scatter, label='Digit')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D Projection of Digits")
plt.savefig("outputs/pca_2d_projection.png", bbox_inches="tight")
#plt.show()
plt.close()

# Yes, Same-digit images tend to cluster together in the 2D space


#------------------------------------PCA Question 3----------------------------------

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(8, 5))

plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.grid(True)
plt.savefig("outputs/pca_variance_explained.png", bbox_inches="tight")
#plt.show()
plt.close()

# From the plot we can tell approximately 13 components are needed to explain 80% of the variance.



#---------------------------------------------PCA Question 4------------------------


def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

n_values = [2, 5, 15, 40]

fig, axes = plt.subplots(len(n_values) + 1, 5, figsize=(10, 10))

for col in range(5):
    axes[0, col].imshow(images[col], cmap="gray_r")
    axes[0, col].set_title(f"Original {y_digits[col]}")
    axes[0, col].axis("off")
    
for row, n in enumerate(n_values, start=1):
    for col in range(5):
        reconstructed = reconstruct_digit(col, scores, pca, n)
        axes[row, col].imshow(reconstructed, cmap="gray_r")
        axes[row, col].axis("off")

        if col == 0:
            axes[row, col].set_ylabel(f"n={n}", fontsize=12)

plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png", bbox_inches="tight")
plt.show()

# when the n = 15 the digits become clearly recognizable
# Yes, this genrally matches the variance curve by capturing most of the variance.