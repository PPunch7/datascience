# Import necessary Lib
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
# %matplotlib inline

## Load dataset
df = pd.read_csv('wk05/fitness_class_2212.csv')

## Display dataset info
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

## Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

## Handle data pattern
# remove text in the 'days_before' colmun
df['days_before'] = df['days_before'].str.extract(r'(\d+)').astype(float)

# replace the 'day_of_week' column to be the same pattern
df['day_of_week'] = df['day_of_week'].str.strip().str[:3].str.capitalize()

## Handle missing values
# Option 1: Drop rows with missing values
# df = df.dropna()

# Option 2: Impute missing values
# using mean value for the columns which are number
numeric_cols = df.select_dtypes(include=[np.number]).columns
imputer_num = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])

# using most frequent value for the columns which include string
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
#df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Check for missing values again
print("\nMissing Values: -> Second time")
print(df.isnull().sum())

# ## Recheck data
# print(df['day_of_week'].unique())
# print(df['time'].unique())
# print(df['category'].unique())

# ------------------------------------------------------------------------------ #
## Assignment: Perform Exploratory Data Analysis
# Remove booking_id from analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'booking_id' in numeric_cols:
    numeric_cols.remove('booking_id')
print("\nNumeric columns used for EDA:", numeric_cols)

# Summary statistics
print("\n=== Descriptive Statistics ===")
print(df[numeric_cols].describe(include='all'))
## All missing numerical values were imputed using the mean strategy,
## while categorical missing values were replaced using the most frequent category.
## This ensured the dataset remained complete for further modelling.

# Distribution of numerical columns (histogram + KDE)
df[numeric_cols].hist(figsize=(12, 10), bins=20, edgecolor='black')
plt.suptitle("Histogram of Numerical Features", fontsize=16)
plt.show()

plt.figure(figsize=(10, 6))
for col in numeric_cols:
    sns.kdeplot(df[col], label=col)
plt.legend()
plt.title("KDE Plots of Numerical Features")
plt.show()
## Histogram and KDE plots suggested that numerical variables such as days_before,
## capacity, and registered show slightly skewed distributions.
## Some features exhibited multiple peaks, indicating the presence of sub-groups.

# Boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplot of Numerical Columns")
plt.xticks(rotation=45)
plt.show()

# Count plots for categorical features
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
n_cat = len(categorical_cols)
n_cols = 2
n_rows = int(np.ceil(n_cat / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
axes = axes.flatten()
for ax, col in zip(axes, categorical_cols):
    sns.countplot(data=df, x=col, ax=ax)
    ax.set_title(f"Count Plot for {col}")
    ax.tick_params(axis='x', rotation=45)
for j in range(len(categorical_cols), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()
## The distribution of day_of_week showed that classes on certain days (e.g. Fri, Thu) were more frequently attended than others.
## Also, most participants was joining during the morning time.
## In terms of categories, the majority of participants were most interested in HIIT.

# Correlation Heatmap
le = LabelEncoder()
df['day_of_week_enc'] = le.fit_transform(df['day_of_week'])
df['category_enc'] = le.fit_transform(df['category'])
df['time_enc'] = le.fit_transform(df['time'])
cols = numeric_cols + ['day_of_week_enc', 'category_enc', 'time_enc']
plt.figure(figsize=(10, 6))
sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
## The heatmap shows that most of the data are not very correlated.
## Only the participation dates and times show a moderate level of correlation.

# Pairplot
feature_cols = [c for c in cols if c != 'attended']
sns.pairplot(df[feature_cols + ['attended']], hue='attended', diag_kind='kde')
plt.show()

# Target class
plt.figure(figsize=(6,4))
sns.countplot(x=df['attended'])
plt.title("Target Class Distribution")
plt.show()
# ------------------------------------------------------------------------------ #

print("\n------------------------------------------------------------------------------------")

## Preprocessing
# Encode categorical variables using label encoding, feel free to try one-hot encoding
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Split features and target
x = df.drop('day_of_week', axis=1)  # Replace 'Attendance' with your target column name
y = df['attended']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Balancing the Dataset (SMOTE: Oversampling using imblearn library)
smote = SMOTE(random_state=42)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train_scaled, y_train)

# Logistic Regression 
lr_balanced = LogisticRegression(random_state=42)
lr_balanced.fit(x_train_balanced, y_train_balanced)
y_pred_lr_balanced = lr_balanced.predict(x_test_scaled)

print("\nLogistic Regression Performance (With Balancing):")
print(classification_report(y_test, y_pred_lr_balanced))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr_balanced))

## Random Forest
rf_balanced = RandomForestClassifier(random_state=42)
rf_balanced.fit(x_train_balanced, y_train_balanced)
y_pred_rf_balanced = rf_balanced.predict(x_test_scaled)

print("\nRandom Forest Performance (With Balancing):")
print(classification_report(y_test, y_pred_rf_balanced))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf_balanced))

# Hyperparameter Tuning (Random Forest)
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5)
grid_search_rf.fit(x_train_balanced, y_train_balanced)

print("\nBest Parameters for Random Forest:")
print(grid_search_rf.best_params_)

# Evaluate tuned Random Forest
best_rf = grid_search_rf.best_estimator_
y_pred_best_rf = best_rf.predict(x_test_scaled)

print("\nTuned Random Forest Performance:")
print(classification_report(y_test, y_pred_best_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best_rf))

# Cross-Validation (Random Forest)
cv_scores_rf = cross_val_score(best_rf, x_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
print("\nRandom Forest Cross-Validation Scores:", cv_scores_rf)
print("Mean CV Accuracy:", cv_scores_rf.mean())

# Gradient Boosting Trees (GBT)
gbt = GradientBoostingClassifier(random_state=42)
gbt.fit(x_train_balanced, y_train_balanced)
y_pred_gbt = gbt.predict(x_test_scaled)

print("\nGradient Boosting Trees Performance:")
print(classification_report(y_test, y_pred_gbt))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_gbt))

# ------------------------------------------------------------------------------ #
## Assignment: 
## 1.Fine-Tune the Hyper-Parameters for Logistic Regression and GBT model
## 2.Use other plots to compare model performance

### Evaluate plot function
def evaluate_with_plot(model_name, y_true, y_pred):
    print(f"\n=== {model_name} ===")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

### Logistic Regressing tuning
param_grid_lr = {
    'penalty': ['l2'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs'],
    'max_iter': [500, 1000],
}
lr_tuned = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid_lr,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
lr_tuned.fit(x_train_balanced, y_train_balanced)
print("\nBest LR Parameters:", lr_tuned.best_params_)

# test set performance
y_pred_lr_tuned = lr_tuned.predict(x_test_scaled)
evaluate_with_plot("Tuned Logistic Regression", y_test, y_pred_lr_tuned)

### Gradient Boosting tuning
param_grid_gbt = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 4],
    'subsample': [0.7, 0.9, 1.0]
}
gbt_tuned = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid_gbt,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
gbt_tuned.fit(x_train_balanced, y_train_balanced)
print("\nBest GBT Parameters:", gbt_tuned.best_params_)

# test set performance
y_pred_gbt_tuned = gbt_tuned.predict(x_test_scaled)
evaluate_with_plot("Tuned Gradient Boosting Trees", y_test, y_pred_gbt_tuned)

### Model performance comparison bar plot
## Collect model metrics
models = {
    "LR_Tuned": y_pred_lr_tuned,
    "RF_Tuned": y_pred_best_rf,
    "GBT_Tuned": y_pred_gbt_tuned
}
performance = {}
for name, pred in models.items():
    performance[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-score": f1_score(y_test, pred)
    }
perf_df = pd.DataFrame(performance).T

# Plot
perf_df.plot(kind='bar', figsize=(10,6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.show()

### ROC curve for LR, RF, GBT
plt.figure(figsize=(8,6))
# LR
fpr, tpr, _ = roc_curve(y_test, lr_tuned.predict_proba(x_test_scaled)[:,1])
plt.plot(fpr, tpr, label="LR Tuned")

# RF
fpr, tpr, _ = roc_curve(y_test, best_rf.predict_proba(x_test_scaled)[:,1])
plt.plot(fpr, tpr, label="RF Tuned")

# GBT
fpr, tpr, _ = roc_curve(y_test, gbt_tuned.predict_proba(x_test_scaled)[:,1])
plt.plot(fpr, tpr, label="GBT Tuned")

plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

### Feature importance plot (RF vs GBT)
feat_names = x.columns

plt.figure(figsize=(10,6))
plt.bar(feat_names, best_rf.feature_importances_)
plt.xticks(rotation=45)
plt.title("Random Forest Feature Importance")
plt.show()

plt.figure(figsize=(10,6))
plt.bar(feat_names, gbt_tuned.best_estimator_.feature_importances_)
plt.xticks(rotation=45)
plt.title("Gradient Boosting Trees Feature Importance")
plt.show()

# ------------------------------------------------------------------------------ #

print("\n------------------------------------------------------------------------------------")

#Bias-Variance Tradeoff (Learning Curves)
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, x, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
    train_sizes, train_scores, test_scores = learning_curve(estimator, x, y, cv=cv, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-Validation Score")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(loc="best")
    plt.show()

# Plot learning curve for Random Forest
plot_learning_curve(best_rf, "Learning Curve (Random Forest)", x_train_balanced, y_train_balanced)

# Plot learning curve for Gradient Boosting Trees
plot_learning_curve(gbt, "Learning Curve (Gradient Boosting Trees)", x_train_balanced, y_train_balanced)
