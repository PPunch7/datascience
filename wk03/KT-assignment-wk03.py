import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### import data ###
df = pd.read_csv('wk03/Diabetes-Classification.csv')
print(df.head())
df.info()
df.isnull().sum()

df=df.dropna()
print(df.head())

# Problems with categorical values in columns:
# 1: these are in string format
# 2: categorical variables confuses

# label encoding
from sklearn.preprocessing import LabelEncoder
cat_col=['Diagnosis']
le=LabelEncoder()

for col in cat_col:
    df[col]=le.fit_transform(df[col])
print(df.head())

#one hot encoding
one_encoded=pd.get_dummies(df, columns=['Gender'])
one_encoded.head()

# how to choose either Label encodeing or onehot encoding:
# it depends on the type of categorical variable:
# Label encoding: when value in features are ordinal: order Ex: Education - Bachelors > Masters > Doctorate
# Onehotending: when categories are not in order (Nominal Values): Gender - Male / Female, Smoking etc.

# we are applying one hot encoding
cat_col=['Gender', 'Exercise', 'Blood Pressure', 'Family History of Diabetes', 'Smoking', 'Diet']
one_encoded=pd.get_dummies(df, columns=cat_col)
one_encoded.head()

# label encoding the one hot encoded columns to convert all the string values in newly generated columns to integers
cat_col=['Gender_Female', 'Gender_Male', 'Exercise_No', 'Exercise_Regular', 
         'Blood Pressure_High', 'Blood Pressure_Low', 'Blood Pressure_Normal',
         'Family History of Diabetes_No', 'Family History of Diabetes_Yes',
         'Smoking_No', 'Smoking_Yes', 'Diet_Healthy', 'Diet_Poor']
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in cat_col:
    one_encoded[col]=le.fit_transform(one_encoded[col])
one_encoded.head()

### ----- Assignment: ----- ###

#Perform -
#Descriptive analysis
#Exploratory data analysis
#here.

#we will discuss about these in the mentoring session

df[['Age','BMI','FBS','HbA1c']].describe()

for col in ['Gender','Blood Pressure','Family History of Diabetes','Smoking','Diet','Exercise','Diagnosis']:
    print(col)
    print(df[col].value_counts(), "\n")

import matplotlib.pyplot as plt
import seaborn as sns

num_cols = ['Age','BMI','FBS','HbA1c']
plt.figure(figsize=(12,8))

for i, col in enumerate(num_cols,1):
    plt.subplot(2,2,i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,8))
for i, col in enumerate(num_cols,1):
    plt.subplot(2,2,i)
    sns.boxplot(x='Diagnosis', y=col, data=df)
    plt.title(f'{col} by Diagnosis')
plt.tight_layout()
plt.show()

cat_cols = ['Gender','Blood Pressure','Family History of Diabetes','Smoking','Diet','Exercise']

plt.figure(figsize=(14,12))
for i, col in enumerate(cat_cols,1):
    plt.subplot(3,2,i)
    sns.countplot(data=df, x=col, hue='Diagnosis')
    plt.xticks(rotation=45)
    plt.title(f'{col} vs Diagnosis')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(one_encoded.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

sns.pairplot(df[['Age','BMI','FBS','HbA1c','Diagnosis']], hue='Diagnosis')
plt.show()

### ------------------------------------------------- ###

# Feature Selection: Removing irrelevant features which does not contribute towards the label(Diagnosis)
# defining feature selection model
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=5)

# splitting label and features
x=one_encoded.drop('Diagnosis', axis=1)
y=one_encoded['Diagnosis']

# applying feature selection
x_selected=selector.fit_transform(x,y)
selected_features=x.columns[selector.get_support()]
print(selected_features)

# created df with selected features
x1=one_encoded[selected_features]

# train test split
from sklearn.model_selection import train_test_split

# for all features
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.4,random_state=10)

# for selected features
#x1_train, x1_test, y_train, y_test=train_test_split(x,y,test_size=0.4,random_state=10)
x1_train, x1_test, y1_train, y1_test=train_test_split(x1,y,test_size=0.4,random_state=10)

# building knn models
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5)
knn_selected=KNeighborsClassifier(n_neighbors=5)

# training models
# all features
knn.fit(x_train,y_train)
y_pred_knn=knn.predict(x_test)

# selcted features
# knn_selected.fit(x1_train,y_train)
knn_selected.fit(x1_train,y1_train)
y_pred_knn1=knn_selected.predict(x1_test)

# classification report
from sklearn.metrics import classification_report

print("nKNN Performance:")
print(classification_report(y_test, y_pred_knn))

print("nKNN_Selected Performance:")
print(classification_report(y_test, y_pred_knn1))

# building svm models
from sklearn.svm import SVC

svm=SVC(kernel='rbf', probability=True,random_state=10)
svm_selected=SVC(kernel='rbf', probability=True,random_state=10)

# training models
# all features
svm.fit(x_train,y_train)
y_pred_svm=svm.predict(x_test)

# selcted features
# svm_selected.fit(x1_train,y_train)
svm_selected.fit(x1_train,y1_train)
y_pred_svm1=svm_selected.predict(x1_test)

# classification report
from sklearn.metrics import classification_report

print("SVM Performance:")
print(classification_report(y_test, y_pred_svm))

print("nSVM_Selected Performance:")
# print(classification_report(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm1))

# confusion: it show labelwise performance of the model
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_svm)
sns.heatmap(cm,annot=True)

# plot ROC curve for svm all features model
from sklearn.metrics import roc_auc_score, roc_curve

y_proba = svm.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
# plt.plot(fpr, tpr, label=f'{'svm'} (AUC = {roc_auc_score(y_test, y_proba):.2f})')
plt.plot(fpr, tpr, label=f'SVM (AUC = {roc_auc_score(y_test, y_proba):.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for SVM(all features) model')
plt.legend()
plt.show()

### ----- Assignment: ----- ###

#1: Plot confusion matrix and ROC curves for other models
## 1.1 Confusion Matrix (KNN / KNN_selected / SVM / SVM_selected)
models = {
    "KNN (all)": (y_test, y_pred_knn),
    "KNN (selected)": (y1_test, y_pred_knn1),
    "SVM (all)": (y_test, y_pred_svm),
    "SVM (selected)": (y1_test, y_pred_svm1),
}

plt.figure(figsize=(10, 10))

for i, (name, (yt, yp)) in enumerate(models.items(), 1):
    cm = confusion_matrix(yt, yp)
    plt.subplot(2, 2, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(name)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.show()

## 1.2 ROC Curve for all models
plt.figure(figsize=(8, 6))

# KNN (all)
y_proba_knn = knn.predict_proba(x_test)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
auc_knn = roc_auc_score(y_test, y_proba_knn)
plt.plot(fpr_knn, tpr_knn, label=f'KNN (all) AUC={auc_knn:.2f}')

# KNN (selected)
y_proba_knn1 = knn_selected.predict_proba(x1_test)[:, 1]
fpr_knn1, tpr_knn1, _ = roc_curve(y1_test, y_proba_knn1)
auc_knn1 = roc_auc_score(y1_test, y_proba_knn1)
plt.plot(fpr_knn1, tpr_knn1, label=f'KNN (sel) AUC={auc_knn1:.2f}')

# SVM (all)
y_proba_svm = svm.predict_proba(x_test)[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
auc_svm = roc_auc_score(y_test, y_proba_svm)
plt.plot(fpr_svm, tpr_svm, label=f'SVM (all) AUC={auc_svm:.2f}')

# SVM (selected)
y_proba_svm1 = svm_selected.predict_proba(x1_test)[:, 1]
fpr_svm1, tpr_svm1, _ = roc_curve(y1_test, y_proba_svm1)
auc_svm1 = roc_auc_score(y1_test, y_proba_svm1)
plt.plot(fpr_svm1, tpr_svm1, label=f'SVM (sel) AUC={auc_svm1:.2f}')

# Random line
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for KNN and SVM models')
plt.legend()
plt.show()


#2: Learn more about evaluation metrics sucha as confusion matrix, accuracy, precision, recall, AUC, ROC etc
#...

#3: Compare the performance(accuracy, f1 score etc) of SVM and KNN models we created using graph plots
# collecting metrics to dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_scores(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

scores = {
    'KNN (all)': get_scores(y_test, y_pred_knn),
    'KNN (sel)': get_scores(y1_test, y_pred_knn1),
    'SVM (all)': get_scores(y_test, y_pred_svm),
    'SVM (sel)': get_scores(y1_test, y_pred_svm1),
}

scores_df = pd.DataFrame(scores).T   # transpose to model rows
print(scores_df)

# Plot bar chart
plt.figure(figsize=(10,6))
scores_df[['accuracy','precision','recall','f1']].plot(kind='bar')
plt.ylim(0, 1)
plt.title('Model performance comparison (KNN vs SVM)')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


#4: Re-do the entire analysis with label encoding instead of one hot encoding and notice performance difference
## Read data and encode all categorical by LabelEncoder (no get_dummies):

# ------------------ Label Encoding Version ------------------ #
df_le = pd.read_csv('wk03/Diabetes-Classification.csv')
df_le = df_le.dropna()

# list of category columns (including Diagnosis)
cat_cols_le = ['Gender', 'Exercise', 'Blood Pressure',
               'Family History of Diabetes', 'Smoking', 'Diet', 'Diagnosis']

le_dict = {}   # sparing for encoder

for col in cat_cols_le:
    le = LabelEncoder()
    df_le[col] = le.fit_transform(df_le[col])
    le_dict[col] = le

X_le = df_le.drop('Diagnosis', axis=1)
y_le = df_le['Diagnosis']

# Feature selection - OLD version
selector_le = SelectKBest(score_func=f_classif, k=5)
X_le_selected = selector_le.fit_transform(X_le, y_le)
selected_features_le = X_le.columns[selector_le.get_support()]
print("Selected features (Label Encoded):", selected_features_le)

Xle1 = df_le[selected_features_le]

# Train-test split
X_le_train, X_le_test, y_le_train, y_le_test = train_test_split(
    X_le, y_le, test_size=0.4, random_state=10, stratify=y_le
)

Xle1_train, Xle1_test, yle1_train, yle1_test = train_test_split(
    Xle1, y_le, test_size=0.4, random_state=10, stratify=y_le
)

# KNN / SVM models
knn_le = KNeighborsClassifier(n_neighbors=5)
knn_le_sel = KNeighborsClassifier(n_neighbors=5)

svm_le = SVC(kernel='rbf', probability=True, random_state=10)
svm_le_sel = SVC(kernel='rbf', probability=True, random_state=10)

# Train & predict
knn_le.fit(X_le_train, y_le_train)
y_pred_knn_le = knn_le.predict(X_le_test)

knn_le_sel.fit(Xle1_train, yle1_train)
y_pred_knn_le1 = knn_le_sel.predict(Xle1_test)

svm_le.fit(X_le_train, y_le_train)
y_pred_svm_le = svm_le.predict(X_le_test)

svm_le_sel.fit(Xle1_train, yle1_train)
y_pred_svm_le1 = svm_le_sel.predict(Xle1_test)

print("\n[KNN LabelEnc] all features:")
print(classification_report(y_le_test, y_pred_knn_le))

print("\n[KNN LabelEnc] selected features:")
print(classification_report(yle1_test, y_pred_knn_le1))

print("\n[SVM LabelEnc] all features:")
print(classification_report(y_le_test, y_pred_svm_le))

print("\n[SVM LabelEnc] selected features:")
print(classification_report(yle1_test, y_pred_svm_le1))


## comparing performance side-by-side
scores_le = {
    'KNN_LE (all)': get_scores(y_le_test, y_pred_knn_le),
    'KNN_LE (sel)': get_scores(yle1_test, y_pred_knn_le1),
    'SVM_LE (all)': get_scores(y_le_test, y_pred_svm_le),
    'SVM_LE (sel)': get_scores(yle1_test, y_pred_svm_le1),
}

scores_all = pd.concat(
    [scores_df, pd.DataFrame(scores_le).T],
    axis=0
)

print(scores_all)

plt.figure(figsize=(10,6))
scores_all[['accuracy','f1']].plot(kind='bar')
plt.ylim(0,1)
plt.title('One-hot vs Label Encoding (KNN & SVM)')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()