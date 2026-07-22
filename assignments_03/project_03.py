import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.tree import DecisionTreeClassifier
from io import BytesIO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    classification_report
)
from sklearn.inspection import DecisionBoundaryDisplay

warnings.filterwarnings("ignore", category=RuntimeWarning)

#-----------------------------Task 1: Load and Explore---------------------------------

COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]



url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
#print(df.head())
print("Total emails:", len(df))

class_counts = df["spam_label"].value_counts()
print("\nClass counts:")
print(class_counts)

class_percentages = df["spam_label"].value_counts(normalize=True) * 100
print("\nClass percentages:")
print(class_percentages)

features = [
    "word_freq_free",
    "char_freq_!",
    "capital_run_length_total"
]

for feature in features:
    plt.figure(figsize=(6, 4))
    ham = df[df["spam_label"] == 0][feature]
    spam = df[df["spam_label"] == 1][feature]
    plt.boxplot([ham, spam], tick_labels=["Ham", "Spam"])
    plt.title(f"{feature}: Ham vs Spam")
    plt.ylabel(feature)
    plt.savefig(f"outputs/{feature}_boxplot.png", bbox_inches="tight")
    plt.show()
    
# Spam emails in genral have higher values for word_freq_free
# there are some overlaps means those features are not suficient on their own to perfectly check ham or spam
    
# many features are skewed towards zero. also the fearure scales vary greatly
# word-frequency and character-frequency features are percentages usually small
# values, while capital letter features can be much larger
# So for distance based models like knn we need to scale it

#----------------------------Task 2: Prepare Your Data--------------------------------


# Dividing the dataset into training and testing parts so we can train the model
# on one part and check its performance on unseen data. Using stratify keeps the
# spam and ham ratio similar in both training and testing sets.

# Scaling the features because different columns have very different value ranges.
# This helps models like KNN, Logistic Regression, and PCA work better by putting
# all features on a similar scale.

# The scaler is fitted only using training data because using test data during
# scaling can cause data leakage.

# PCA is applied only after scaling because PCA depends on feature variance and
# larger-scale features can dominate the components. PCA is also fitted only on
# training data to prevent information from the test set affecting the model.


X = df.drop("spam_label", axis=1)
y = df["spam_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA()
pca.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n = np.argmax(cumulative_variance >= 0.90) + 1

print("Number of components where variance = 90 %: ", n)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.grid(True)
plt.savefig("outputs/pca_variance_explained.png", bbox_inches="tight")
plt.show()
plt.close()

X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]

print("Original scaled shape:", X_train_scaled.shape)
print("PCA-reduced shape:", X_train_pca.shape)


#-------------------------------Task 3: A Classifier Comparison---------------------------------

knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train,y_train)
y_pred_knn_unscaled = knn_unscaled.predict(X_test)
print("- " * 30)
print("KNN Unscaled Accuracy:", accuracy_score(y_test, y_pred_knn_unscaled))
print("Unscaled Classification report")
print(classification_report(y_test, y_pred_knn_unscaled))


knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled,y_train)
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)
print("- " * 30)
print("KNN Scaled Accuracy:", accuracy_score(y_test, y_pred_knn_scaled))
print("Scaled Classification report")
print(classification_report(y_test, y_pred_knn_scaled))


knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca,y_train)
y_pred_knn_pca = knn_pca.predict(X_test_pca)
print("- " * 30)
print("KNN PCA Accuracy:", accuracy_score(y_test, y_pred_knn_pca))
print("KNN PCA Classification report")
print(classification_report(y_test, y_pred_knn_pca))


depth_values = [3, 5, 10, None]

for depth in depth_values:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train,y_train)
    train_accuracy = dt.score(X_train, y_train)
    test_accuracy = dt.score(X_test, y_test)
    print(f"Depth {depth}: Train Accuracy={train_accuracy:.5f}, Test Accuracy={test_accuracy:.5f}")
    
# As depth increases i notice a increase in training accuracy as the the tree
# becomes more complex and learns more from the training data.
# If training accuracy is much higher than test accuracy, it means overfitting,
# so in production i will choose depth 5 as the traiining accuracy and the test accuracy are closer.

dt_fixed = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_fixed.fit(X_train, y_train)
y_pred_dt_fixed = dt_fixed.predict(X_test)

print("- " * 30)
print("Fixed Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt_fixed))
print("Fixed Decision Tree Classification report")
print(classification_report(y_test, y_pred_dt_fixed))


#---------------------------- Random Forest ------------------------------------------

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("- " * 30)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification report")
print(classification_report(y_test, y_pred_rf))


logi_scaled = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
logi_scaled.fit(X_train_scaled, y_train)
y_pred_log_scaled = logi_scaled.predict(X_test_scaled)
print("- " * 30)
print("Logistic Regression Scaled Accuracy: ", accuracy_score(y_test, y_pred_log_scaled))
print("Logistic Regression Scaled Classification report")
print(classification_report(y_test, y_pred_log_scaled))

logi_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
logi_pca.fit(X_train_pca, y_train)
y_pred_logi_pca = logi_pca.predict(X_test_pca)
print("- " * 30)
print("Logistic Regression PCA Accuracy:",accuracy_score(y_test, y_pred_logi_pca))
print("Logistic Regression PCA Classification report")
print(classification_report(y_test, y_pred_logi_pca))


# From the results, Random Forest performed the best with the highest accuracy (94.46%), followed by Logistic Regression (92.94%).
# KNN and Decision Tree also gave good results but were slightly less accurate.

# For both KNN and Logistic Regression, the non-PCA models performed slightly better than the PCA versions. 
# This matches my hypothesis that PCA would reduce the number of features and make the model simpler, 
# but it could also lose some useful information.

# for a spam filter, accuracy is not the most important metric. I would rather minimize false positives because marking
# a genuine email as spam is more harmful than letting a spam email reach the inbox. Missing an important email can 
# cause bigger problems than deleting a spam message.


best_model = rf
best_predictions = y_pred_rf

ConfusionMatrixDisplay.from_predictions( y_test, best_predictions)
plt.savefig( "outputs/best_model_confusion_matrix.png", bbox_inches="tight")
#plt.show()
plt.close()

rf_importances = rf.feature_importances_
rf_indices = np.argsort(rf_importances)[-10:]
rf_top_features = X.columns[rf_indices]
print("- " * 30)
print("Top 10 Random Forest Features:")
print(rf_top_features)

plt.figure(figsize=(8,5))
plt.barh(rf_top_features, rf_importances[rf_indices])
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.savefig("outputs/feature_importances.png", bbox_inches="tight")
plt.show()
plt.close()

#Both models mostly agree on the important features. Words like "free," "remove," "you," and "your," along with $, !, 
# and lots of capital letters, are strong indicators of spam. This matches what we'd expect since spam emails often 
# use attention-grabbing language and formatting.



#---------------------------------------Task 4: Cross-Validation-----------------------------------------

all_models = {
    "KNN Unscaled": (KNeighborsClassifier(n_neighbors=5), X_train,y_train ),
    "KNN Scaled": (KNeighborsClassifier(n_neighbors=5), X_train_scaled, y_train),
    "KNN PCA": (KNeighborsClassifier(n_neighbors=5),X_train_pca, y_train),
    "Decision Tree (Depth=5)": (DecisionTreeClassifier(max_depth=5, random_state=42), X_train, y_train),
    "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42),X_train, y_train),
    "Logistic Regression Scaled": (LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"), X_train_scaled, y_train),
    "Logistic Regression PCA": (LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),X_train_pca, y_train)
}

results = {}

for name, (model, X_data, y_data) in all_models.items():
    scores = cross_val_score(model, X_data, y_data, cv=5, scoring="accuracy")
    
    results[name] = (scores.mean(), scores.std())

    print(f"{name}")
    print(f"Mean Accuracy : {scores.mean():.5f}")
    print(f"Std Deviation : {scores.std():.5f}")
    print("- " * 30)

best_accuracy = max(results.items(), key=lambda x: x[1][0])
best_stability = min(results.items(), key=lambda x: x[1][1])

print(f"\nMost Accurate Model : {best_accuracy[0]} ({best_accuracy[1][0]:.5f})")
print(f"Most Stable Model   : {best_stability[0]} ({best_stability[1][1]:.5f})")


#-------------------------Build your pipelines------------------------------------


rf_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

rf_pipeline.fit(X_train, y_train)
rf_pipeline_predictions = rf_pipeline.predict(X_test)
print("- " * 30)
print("Random Forest Pipeline Classification Report")
print(classification_report(y_test, rf_pipeline_predictions))

logistic_pipeline.fit(X_train, y_train)
logistic_pipeline_predictions = logistic_pipeline.predict(X_test)

print("- " * 30)
print("Logistic Regression Pipeline Classification Report")
print(classification_report(y_test, logistic_pipeline_predictions))


print("- " * 30)
print("Random Forest Pipeline Accuracy:",  rf_pipeline.score(X_test, y_test))
print("Logistic Regression Pipeline Accuracy:", logistic_pipeline.score(X_test, y_test))

# After checking the result i Confirm the results match my earlier manual approach
# I did not include PCA because Logistic Regression with PCA performed worse than the scaled version.

# The pipelines have different structures because different models need different preprocessing steps.
# Random Forest can work directly with the original features, while Logistic Regression performs better in scaled features

# The main benefit of using pipelines is that they combine preprocessing and the model together into one object.
# This reduces mistakes like a forgotton steps, prevents data leakage, and makes the model easier to share and deploy.