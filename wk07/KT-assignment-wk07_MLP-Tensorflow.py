# Import necessary lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#%matplotlib inline

# Load the dataset
df = pd.read_csv('wk07/Diabetes-Classification.csv')

# Display dataset info
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values by dropping rows
df = df.dropna()

# Check column names
print("\nColumn Names:")
print(df.columns)

## Preprocessing ##
# Identify categorical and numerical columns
categorical_cols = ['Gender', 'Blood Pressure', 'Family History of Diabetes', 'Smoking', 'Diet', 'Exercise']
numerical_cols = ['Age', 'BMI', 'FBS', 'HbA1c']

# Target column
target_column = 'Diagnosis'

# Create a ColumnTransformer to preprocess the data in the columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

## Split features and target ##
x = df.drop(target_column, axis=1)
y = df[target_column]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Apply preprocessing
x_train_scaled = preprocessor.fit_transform(x_train)
x_test_scaled = preprocessor.transform(x_test)

# Convert y_train and y_test to binary (label encoding)
y_train = y_train.map({'No': 0, 'Yes': 1}).values
y_test = y_test.map({'No': 0, 'Yes': 1}).values

# Convert x_train_scaled and x_test_scaled to NumPy arrays 
# (Tensorflow does not accept pandas dataframe as input, it accepts either Tensorflow tensors or NumPy arrays)
x_train_scaled = x_train_scaled.astype('float32')
x_test_scaled = x_test_scaled.astype('float32')

# ------------------------------------------------------------------ #

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Input

# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Build the TensorFlow MLP model
tf_model = Sequential([
    Input(shape=(x_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = tf_model.fit(x_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
# y_pred_tf = (tf_model.predict(x_test_scaled) > 0.5).astype(int)
y_proba_tf = tf_model.predict(x_test_scaled).ravel()
y_pred_tf = (y_proba_tf > 0.5).astype(int)
print("\nTensorFlow MLP Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_tf))
print("Classification Report:")
print(classification_report(y_test, y_pred_tf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tf))

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (TensorFlow)')
plt.legend()
plt.show()

# ------------------------------------------------------------------ #
### Assignment: 
## Check and deal with the dataset imbalance 
# Check imbalance
unique, counts = np.unique(y_train, return_counts=True)
class_dist = dict(zip(unique, counts))
print("\nTrain class distribution:", class_dist)
print("Train positive ratio:", class_dist.get(1,0) / len(y_train))

from sklearn.utils.class_weight import compute_class_weight
classes = np.array([0, 1])
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print("\nClass weights:", class_weight_dict)
history = tf_model.fit(
    x_train_scaled, y_train,
    epochs=50, batch_size=32,
    validation_split=0.2,
    class_weight=class_weight_dict,
    verbose=1
)

## Fine-Tune the above neural network by changing the values of hyperparameters and report findings
# Create build model function
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
def build_mlp(input_dim,
              n1=64, n2=32,
              dropout1=0.2, dropout2=0.2,
              lr=1e-3):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(n1, activation='relu'),
        Dropout(dropout1),
        Dense(n2, activation='relu'),
        Dropout(dropout2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Callbacks to avoid overfitting
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5)
]

# Hyperparameter tuning
from sklearn.metrics import roc_auc_score, f1_score
configs = [
    {"n1": 64, "n2": 32, "dropout1": 0.2, "dropout2": 0.2, "lr": 1e-3, "batch_size": 32},
    {"n1": 128, "n2": 64, "dropout1": 0.3, "dropout2": 0.3, "lr": 1e-3, "batch_size": 32},
    {"n1": 64, "n2": 32, "dropout1": 0.1, "dropout2": 0.1, "lr": 5e-4, "batch_size": 32},
    {"n1": 128, "n2": 32, "dropout1": 0.2, "dropout2": 0.2, "lr": 5e-4, "batch_size": 64},
]

results = []
best_model = None
best_auc = -np.inf

for i, cfg in enumerate(configs, start=1):
    print(f"\n--- Tuning run {i}/{len(configs)}: {cfg} ---")
    model = build_mlp(
        input_dim=x_train_scaled.shape[1],
        n1=cfg["n1"], n2=cfg["n2"],
        dropout1=cfg["dropout1"], dropout2=cfg["dropout2"],
        lr=cfg["lr"]
    )
    hist = model.fit(
        x_train_scaled, y_train,
        epochs=80,
        batch_size=cfg["batch_size"],
        validation_split=0.2,
        class_weight=class_weight_dict,        # for editting imbalance
        callbacks=callbacks,
        verbose=0
    )

    y_proba = model.predict(x_test_scaled).ravel()
    y_pred = (y_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    results.append({
        **cfg, 
        "accuracy": acc, 
        "roc_auc": auc, 
        "f1": f1, 
        "best_val_loss": min(hist.history["val_loss"])
    })
    print(f"Accuracy={acc:.4f} | ROC_AUC={auc:.4f} | F1={f1:.4f} | best_val_loss={min(hist.history['val_loss']):.4f}")

    # Collect the best model
    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_cfg = cfg

results_df = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)
print("\n=== Tuning Summary (sorted by ROC AUC) ===")
print(results_df)

print("\nBest model configuration:")
print(best_cfg)
print("Best ROC AUC:", best_auc)

## Use other evaluation metrics and plots to visualise performance of the model
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

# Confusion Matrix Heatmap + ROC + PR Curve
y_proba = best_model.predict(x_test_scaled).ravel()

# ROC AUC / PR AUC
roc_auc = roc_auc_score(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
print("\nROC AUC:", roc_auc)
print("PR AUC (Average Precision):", ap)

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.tight_layout()
plt.show()

# PR curve
prec, rec, thr = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6,4))
plt.plot(rec, prec)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.show()

# Choose approximately threshold
thresholds = np.linspace(0.05, 0.95, 19)
f1s = []

for t in thresholds:
    y_pred_t = (y_proba > t).astype(int)
    f1s.append(f1_score(y_test, y_pred_t, zero_division=0))

best_t = thresholds[int(np.argmax(f1s))]
print("\nBest threshold (by F1):", best_t)

plt.figure(figsize=(6,4))
plt.plot(thresholds, f1s, marker='o')
plt.title("F1-score vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("F1-score")
plt.tight_layout()
plt.show()

# Evaluate at best threshold
y_pred_best_t = (y_proba > best_t).astype(int)
print("\nClassification report @ best threshold:")
print(classification_report(y_test, y_pred_best_t))
cm = confusion_matrix(y_test, y_pred_best_t)

plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix @ Best Threshold")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Training curves
plt.figure(figsize=(10,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()