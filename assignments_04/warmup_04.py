import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
    f1_score
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#---------------------------------ROC and AUC-------------------------------

#----------------------------ROC Question 1----------------------------------

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train,y_train)

lr_probs =lr.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)

print(f"Logistic Regression AUC Score : {lr_auc}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled,y_train)
knn_probs = knn.predict_proba(X_test_scaled)[:, 1]
knn_auc = roc_auc_score(y_test, knn_probs)

print(f"KNN AUC Score : {knn_auc}")

# After comparing the both AUC the knn auc value is more (0.9394)
# that indicates the knn classifier model better separates the two classes
# Since AUC is calculated over all possible classification thresholds, KNN performed better overall


#--------------------------------ROC Question 2-----------------------------------

fpr_lr, tpr_lr , _ = roc_curve(y_test, lr_probs)

knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)

fig, ax = plt.subplots(figsize=(6, 5))

RocCurveDisplay(fpr=fpr_lr, tpr=tpr_lr).plot(
    ax=ax,
    name=f"Logistic Regression (AUC={lr_auc:.3f})"
)

RocCurveDisplay(fpr=knn_fpr, tpr=knn_tpr).plot(
    ax=ax,
    name=f"KNN k=5 (AUC={knn_auc:.3f})"
)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")

ax.set_title("ROC Curve Comparison")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/roc_comparison.png")
#plt.show()
plt.close()

# after analyzing we can see the the knn model reaches TPR = 0.80 with FPR = 0.06 approx
# the Logistic Regression model reaches TPR = 0.80 with FPR = 0.57 approx
# the KNN model has lower FPR and that means that if we needed to catch 80% of the positive the knn model will raise 
# lower false alarm approx 6% of the time compared to Logistic Regression model which will raise approx 57% of the time.


#------------------------------------------------ROC Question 3----------------------------------------------

y_probs_lr = lr.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs_lr)

best_f1 = 0
best_threshold = 0
best_tpr = 0
best_fpr = 0

for threshold, tpr_value, fpr_value in zip(thresholds, tpr, fpr):
    y_pred = (y_probs_lr >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_tpr = tpr_value
        best_fpr = fpr_value
        
print(f"Best threshold : {best_threshold:.4f}")
print(f"Best TPR (Recall) : {best_tpr:.4f}")
print(f"Best FPR: {best_fpr:.4f}")
print(f"Best F1 Score: {best_f1:.4f}")

# The optimal threshold (0.2757) compare to the default 0.5 is significantly lower which 
# means the model prioritizes identifying more positive cases which means it will let a spam let come to the inbox 
# than letting a real email flag as spam
# I choose a threshold lower than 0.5 when missing a value is more costly than getting a wrong flag as example Fraud Detection


#--------------------------------GridSearchCV------------------------------------

##-------------------------------GridSearch Question 1---------------------------

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
])

param_grid = {
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}


grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

print(f"GridSearchCV LR Best C:      {grid_search.best_params_['clf__C']}")
print(f"GridSearchCV LR Best CV AUC: {grid_search.best_score_:.4f}")

best_pipe_lr = grid_search.best_estimator_

y_pred  = best_pipe_lr.predict(X_test)
y_probs = best_pipe_lr.predict_proba(X_test)[:, 1]
print(f"Test AUC: {roc_auc_score(y_test, y_probs):.4f}")


default_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
])

default_pipe.fit(X_train, y_train)

default_probs = default_pipe.predict_proba(X_test)[:, 1]
default_auc = roc_auc_score(y_test, default_probs)
test_auc = roc_auc_score(y_test, y_probs)
print(f"Default C=1.0 Test AUC: {default_auc:.4f}")
print(f"Change in Test AUC: {test_auc - default_auc:.4f}")


#--------------------------------GridSearch Question 2--------------------------------

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", DecisionTreeClassifier(random_state=42))
])

param_grid = {
    "clf__max_depth": [2, 3, 5, 8, None]
}

grid_search_dt = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search_dt.fit(X_train, y_train)

print(f"Best max_depth DT: {grid_search_dt.best_params_['clf__max_depth']}")
print(f"Best CV AUC DT: {grid_search_dt.best_score_:.4f}")

best_pipe_dt = grid_search_dt.best_estimator_
y_probs_dt = best_pipe_dt.predict_proba(X_test)[:, 1]
test_auc_dt = roc_auc_score(y_test, y_probs_dt)
print(f"Test AUC: {test_auc:.4f}")

# As the AUC score came same (0.7057) so neither out performed the other. 
# I would choose logistic regression for further development because it is simpler,
# more interpretable, and generally less prone to overfitting
# AUC is not the only thing to consider. I would also consider precision, recall, F1 score.

#---------------------------------------GridSearch Question 3---------------------------------------

results = pd.DataFrame(grid_search.cv_results_)
print("Question 4 ")
#print(results.head(5))
results = results[["param_clf__C", "mean_test_score", "std_test_score"]].sort_values(by="mean_test_score", ascending=False)
print(results.to_string(index=False))

# C = 100.0 and C = 10.0 have nearly identical mean CV AUC scores (0.772680 vs. 0.772649) 
# and  C = 100.0 has a slightly lower standard deviation (0.005669 vs. 0.005743).
# I would choose C = 100 because even if the mean nearly identical the standard deviation is lower.



#-----------------------------------------joblib------------------------------

#----------------------------joblib Question 1--------------------------


joblib.dump(best_pipe_lr, "models/warmup_model.pkl")
print("Model Saved")

# --- Simulated prediction script ---.

loaded_clf = joblib.load("models/warmup_model.pkl")

original_preds = best_pipe_lr.predict(X_test)
loaded_preds   = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")

# As our saved model is trained on scaled data if we save 
# logistic regression model (without the scaler) and then called .predict(X_test)fit 
# it with unscaled data the predcion results wont be accurate .


#-----------------------------------joblib Question 2------------------------------------

new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

loaded_model_pred = loaded_clf.predict(new_samples)
loaded_model_proba = loaded_clf.predict_proba(new_samples)

for i, (pred, prob) in enumerate(zip(loaded_model_pred, loaded_model_proba), start=1):
    print(f"Sample {i}:")
    print(f"  Predicted class: {pred}")
    print(f"  Probability of class 1: {prob[1]:.4f}")
    
# I expect the all-zeros row will be predcited as the baseline of the model mostlikely i guess class 0
# because every feature has the same value after the pipeline applies the StandardScaler