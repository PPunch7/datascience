# Import necessary lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    precision_score, recall_score, f1_score, log_loss, roc_auc_score, roc_curve
)
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn import tree
# %matplotlib inline

# Load the dataset
df = pd.read_csv('wk04/IRIS.csv')

# Display dataset info
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check number of classes in label
print("\nCheck unique in species")
print(pd.unique(df['species']))

# Preprocessing
# Encode target variable
if df['species'].dtype == 'object':
    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])

# Check uncoded classes
print("Encode of unique in species")
print(pd.unique(df['species']))

# Check for dataset imbalance
print('\nNumber of samples in each class:')
print(df['species'].value_counts())

### Balancing dataset ###
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(df['species']), y=df['species'])
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(df['species']), class_weights)}

# Compute sample weights for each instance
sample_weights = compute_sample_weight(class_weight=class_weight_dict, y=df['species'])
print(class_weight_dict)

# Split features and target
x = df.drop('species', axis=1)
y = df['species']

# Feature scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Train-test split
x_train, x_test, y_train, y_test, sw_train, sw_test = train_test_split(
    x_scaled, y, sample_weights, test_size=0.3, stratify=y, random_state=42)

# ------------------------------------------------------------------------------ #
# Assignment: perform exploratory data analysis using decriptive statistics and different plot. Note down and Discuss 

# Summary statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Plot distribution of each numerical feature
df_features = df.drop('species', axis=1)
df_features.hist(figsize=(10, 8), bins=15, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=14)
plt.show()
## The descriptive statistics indicate that all numerical features (sepal length, sepal width, petal length, and petal width) fall within reasonable biological ranges.
## The dataset contains no missing values, and each feature exhibits distinct means and standard deviations, suggesting variability across species groups.

# KDE plot
plt.figure(figsize=(12, 8))
for col in df_features.columns:
    sns.kdeplot(df_features[col], label=col)
plt.legend()
plt.title("KDE of Numerical Features")
plt.show()
## Histogram and KDE plots reveal that most features follow moderately normal or slightly skewed distributions.
## Petal length and petal width show clear multimodal behaviour, consistent with well-separated species clusters.
## Sepal width shows the greatest spread, while petal width exhibits tighter grouping.

# Boxplots
plt.figure(figsize=(10, 7))
df_features.boxplot()
plt.title("Boxplot of Features")
plt.show()
## Boxplots indicate minimal outliers across features.
## Sepal width shows minor deviations, which is expected due to natural biological variation.
## No significant anomalies requiring removal were detected.

# Pairplot
sns.pairplot(df, hue="species", diag_kind="kde")
plt.suptitle("Pairplot of IRIS Dataset", y=1.02)
plt.show()
## Pairplot analysis shows strong linear separation between species based on petal length and petal width.
## Setosa is clearly distinguishable from Versicolour and Virginica, forming a compact cluster.
## Versicolour and Virginica display partial overlap, which aligns with known classification difficulty.

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
## The correlation heatmap demonstrates:
## - Strong positive correlation between petal length and petal width (r ≈ 0.97)
## - Moderate correlation between sepal and petal dimensions
## - Weak correlation between sepal length and sepal width
## These correlations suggest that petal-based features are the most influential predictors for species classification.

# Class Distribution Visualisation
plt.figure(figsize=(6, 4))
sns.countplot(x=df['species'])
plt.title("Class Distribution of Species")
plt.show()
## Class distribution is evenly balanced across the three species categories (Setosa, Versicolour, Virginica).
## Therefore, the dataset does not suffer from imbalance issues, and weighting adjustments are optional rather than essential.

# ------------------------------------------------------------------------------ #
print("\n------------------------------------------------------------------------------------")

### Multi-Class Classification without dataset balancing
## Logistic Regression (One-vs-Rest)
lr_ovr = OneVsRestClassifier(LogisticRegression())
lr_ovr.fit(x_train, y_train)
y_pred_lr_ovr = lr_ovr.predict(x_test)

print("\nLogistic Regression (One-vs-Rest) Performance:")
print(classification_report(y_test, y_pred_lr_ovr))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr_ovr))

from sklearn import set_config
# Enable metadata routing for OneVsRestClassifier
set_config(enable_metadata_routing=True)

# Train model with sample weights using OneVsRestClassifier with Logistic Regression
clf = OneVsRestClassifier(LogisticRegression(class_weight=class_weight_dict, max_iter=10000))
clf.estimator.set_fit_request(sample_weight=True)  # Explicitly request sample_weight
clf.fit(x_train, y_train, sample_weight=sw_train)

y_pred = clf.predict(x_test)

## Classification Report and Confusion Matrix
print("\nLogistic Regression (One-vs-Rest) Performance balanced weights:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

## Logistic Regression (One-vs-One)
lr_ovo = OneVsOneClassifier(LogisticRegression())
lr_ovo.fit(x_train, y_train)
y_pred_lr_ovo = lr_ovo.predict(x_test)

print("\nLogistic Regression (One-vs-One) Performance:")
print(classification_report(y_test, y_pred_lr_ovo))

## Softmax Regression (Multinomial Logistic Regression)
softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs')
softmax.fit(x_train, y_train)
y_pred_softmax = softmax.predict(x_test)

print("\nSoftmax Regression Performance:")
print(classification_report(y_test, y_pred_softmax))

#### Evaluation Metrics ####
def evaluate_model(y_true, y_pred, y_proba=None):
    print("--- START ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    if y_proba is not None:
        print("Log Loss:", log_loss(y_true, y_proba))
        print("ROC AUC Score (OvR):", roc_auc_score(y_true, y_proba, multi_class='ovr'))
    print("--- END ---")

## Evaluate Logistic Regression (OvR)
print("\nEvaluate Logistic Regression (OvR) Performance:")
evaluate_model(y_test, y_pred_lr_ovr, lr_ovr.predict_proba(x_test))

## Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
print("\nDecision Tree Performance:")
evaluate_model(y_test, y_pred_dt, dt.predict_proba(x_test))

## Visualize Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt, filled=True, feature_names=x.columns, class_names=le.classes_)
plt.show()

## Hyperparameter Tuning (Decision Tree)
param_grid = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(x_train, y_train)
print("\nBest Parameters for Decision Tree:")
print(grid_search.best_params_)

## Evaluate tuned Decision Tree
best_dt = grid_search.best_estimator_
y_pred_best_dt = best_dt.predict(x_test)
print("\nTuned Decision Tree Performance:")
evaluate_model(y_test, y_pred_best_dt, best_dt.predict_proba(x_test))

# ------------------------------------------------------------------------------ #
# Assignment: try all the other models with balancing the dataset using weights and note down the findings
# plot confusion matrices for all models

## Functions for evaluation and confusion matrix plotting
class_names = le.classes_  # using species to be the label
def evaluate_and_plot(name, y_true, y_pred, y_proba=None):
    print("--- START ---")
    print(f"=== {name} ===")
    print("Confusion Matrix (numbers):")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    if y_proba is not None:
        try:
            print("Log Loss:", log_loss(y_true, y_proba))
            print("ROC AUC (OvR):", roc_auc_score(y_true, y_proba, multi_class='ovr'))
        except Exception as e:
            print("Could not compute log loss / ROC AUC:", e)

    # Plot confusion matrix
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(name)
    plt.tight_layout()
    plt.show()
    print("--- END ---")

## Logistic Regression (One-Vs-Rest) -> unweighted vs weighted
print("\n------------------------------------------------------------------------------------")
print("Logistic Regression Models")

# Logistic Regression (One-vs-Rest) -> unweighted
lr_ovr = OneVsRestClassifier(LogisticRegression(
    max_iter=10000, 
    multi_class='auto'
))
lr_ovr.fit(x_train, y_train)
y_pred_lr_ovr = lr_ovr.predict(x_test)
y_proba_lr_ovr = lr_ovr.predict_proba(x_test)
evaluate_and_plot("LR OvR (unweighted)", y_test, y_pred_lr_ovr, y_proba_lr_ovr)

# Logistic Regression (One-vs-Rest) -> weighted with class_weight
lr_ovr_w = OneVsRestClassifier(LogisticRegression(
    max_iter=10000,
    multi_class='auto',
    class_weight=class_weight_dict
))
lr_ovr_w.fit(x_train, y_train)
y_pred_lr_ovr_w = lr_ovr_w.predict(x_test)
y_proba_lr_ovr_w = lr_ovr_w.predict_proba(x_test)
evaluate_and_plot("LR OvR (weighted class_weight)", y_test, y_pred_lr_ovr_w, y_proba_lr_ovr_w)

# Logistic Regression (One-vs-One) -> unweighted
lr_ovo = OneVsOneClassifier(LogisticRegression(
    max_iter=10000
))
lr_ovo.fit(x_train, y_train)
y_pred_lr_ovo = lr_ovo.predict(x_test)
# y_proba_lr_ovo = lr_ovo.predict_proba(x_test)
evaluate_and_plot("LR OvO (unweighted)", y_test, y_pred_lr_ovo)

# Logistic Regression (One-vs-One) -> weighted
lr_ovo_w = OneVsOneClassifier(LogisticRegression(
    max_iter=10000,
    class_weight=class_weight_dict
))
lr_ovo_w.fit(x_train, y_train)
y_pred_lr_ovo_w = lr_ovo_w.predict(x_test)
# y_proba_lr_ovo_w = lr_ovo_w.predict_proba(x_test)
evaluate_and_plot("LR OvO (weighted class_weight)", y_test, y_pred_lr_ovo_w)

# Softmax Regression (Multinomial) -> unweighted
softmax = LogisticRegression(
    multi_class='multinomial', 
    solver='lbfgs', 
    max_iter=10000
)
softmax.fit(x_train, y_train)
y_pred_softmax = softmax.predict(x_test)
y_proba_softmax = softmax.predict_proba(x_test)
evaluate_and_plot("Softmax (unweighted)", y_test, y_pred_softmax, y_proba_softmax)

# Softmax Regression (Multinomial) -> weighted
softmax_w = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=10000,
    class_weight=class_weight_dict
)
softmax_w.fit(x_train, y_train)
y_pred_softmax_w = softmax_w.predict(x_test)
y_proba_softmax_w = softmax_w.predict_proba(x_test)
evaluate_and_plot("Softmax (weighted class_weight)", y_test, y_pred_softmax_w, y_proba_softmax_w)

## Decition Tree -> unweighted vs weighted
print("\n------------------------------------------------------------------------------------")
print("Decision Tree Models")

# Decision Tree - Unweighted
dt = DecisionTreeClassifier(
    random_state=42
)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
y_proba_dt = dt.predict_proba(x_test)
evaluate_and_plot("Decision Tree (unweighted)", y_test, y_pred_dt, y_proba_dt)

# Decision Tree - Weighted
dt_w = DecisionTreeClassifier(
    random_state=42, 
    class_weight=class_weight_dict
)
dt_w.fit(x_train, y_train)
y_pred_dt_w = dt_w.predict(x_test)
y_proba_dt_w = dt_w.predict_proba(x_test)
evaluate_and_plot("Decision Tree (weighted class_weight)", y_test, y_pred_dt_w, y_proba_dt_w)

# Visualise one of the trees (e.g. weighted)
plt.figure(figsize=(16, 8))
plot_tree(dt_w, filled=True, feature_names=x.columns, class_names=class_names)
plt.title("Weighted Decision Tree")
plt.show()

## Random Forest -> unweighted vs weighted
print("\n------------------------------------------------------------------------------------")
print("Random Forest Models")

from sklearn.ensemble import RandomForestClassifier

# Random Forest - Unweighted
rf = RandomForestClassifier(
    n_estimators=100, 
    random_state=42
)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
y_proba_rf = rf.predict_proba(x_test)
evaluate_and_plot("Random Forest (unweighted)", y_test, y_pred_rf, y_proba_rf)

# Random Forest - Weighted
rf_w = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight=class_weight_dict
)
rf_w.fit(x_train, y_train)
y_pred_rf_w = rf_w.predict(x_test)
y_proba_rf_w = rf_w.predict_proba(x_test)
evaluate_and_plot("Random Forest (weighted class_weight)", y_test, y_pred_rf_w, y_proba_rf_w)

## Gradient Boosting -> unweighted vs sample_weight
print("\n------------------------------------------------------------------------------------")
print("Gradient Boosting Models")

from sklearn.ensemble import GradientBoostingClassifier

# Gradient Boosting - Unweighted
gb = GradientBoostingClassifier(random_state=42)
gb.fit(x_train, y_train)
y_pred_gb = gb.predict(x_test)
y_proba_gb = gb.predict_proba(x_test)
evaluate_and_plot("Gradient Boosting (unweighted)", y_test, y_pred_gb, y_proba_gb)

# Gradient Boosting - Weighted using sample_weight
gb_w = GradientBoostingClassifier(random_state=42)
gb_w.fit(x_train, y_train, sample_weight=sw_train)
y_pred_gb_w = gb_w.predict(x_test)
y_proba_gb_w = gb_w.predict_proba(x_test)
evaluate_and_plot("Gradient Boosting (weighted sample_weight)", y_test, y_pred_gb_w, y_proba_gb_w)

# ------------------------------------------------------------------------------ #
## Applying class weights and sample weights did not substantially change model performance, 
## which is consistent with the fact that the Iris dataset is intrinsically balanced. 
## Minor variations in recall and precision across species classes were observed 
## but did not indicate a systematic improvement due to weighting.
# ------------------------------------------------------------------------------ #
