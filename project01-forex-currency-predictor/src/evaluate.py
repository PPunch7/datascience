import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

# =====================================
# Regression metrics
# =====================================
def mae(y, yhat):
    return np.mean(np.abs(y - yhat))

def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat) ** 2))

def mape(y, yhat):
    return np.mean(np.abs((y - yhat) / y)) * 100

def print_metrics(name, y, yhat):
    print(f"\n{name}")
    print("MAE :", mae(y, yhat))
    print("RMSE:", rmse(y, yhat))
    print("MAPE:", mape(y, yhat))

def collect_metrics(name, y, yhat):
    return {
        "model": name,
        "MAE": mae(y, yhat),
        "RMSE": rmse(y, yhat),
        "MAPE": mape(y, yhat)
    }

# =====================================
# Direction metrics
# =====================================
def to_direction(arr):
    # convert series -> direction
    # 1 = up, 0 = down or stable
    diff = np.diff(arr, prepend=arr[0])
    return (diff > 0).astype(int)

def direction_accuracy(y_true, y_pred):
    d_true = to_direction(y_true)
    d_pred = to_direction(y_pred)

    return np.mean(d_true == d_pred)

def print_direction_report(name, y_true, y_pred):
    d_true = to_direction(y_true)
    d_pred = to_direction(y_pred)

    print(f"\n{name} — Direction Accuracy:",
          direction_accuracy(y_true, y_pred))

    print("\nConfusion Matrix (Direction):")
    print(confusion_matrix(d_true, d_pred))

    print("\nClassification Report:")
    print(classification_report(d_true, d_pred, digits=4))

def plot_confusion(y_true, y_pred, name):
    d_true = to_direction(y_true)
    d_pred = to_direction(y_pred)

    cm = confusion_matrix(d_true, d_pred)

    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Direction Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.show()

# =====================================
# Multi-model direction evaluation
# =====================================
def plot_confusion_multi(model_preds: dict, y_true):
    """
    model_preds = {
        "xgb": y_pred_xgb,
        "lgb": y_pred_lgb,
        ...
    }
    """
    d_true = to_direction(y_true)

    n = len(model_preds)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten()

    for ax, (name, y_pred) in zip(axes, model_preds.items()):

        d_pred = to_direction(y_pred)
        cm = confusion_matrix(d_true, d_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar=False
        )

        acc = direction_accuracy(y_true, y_pred)

        ax.set_title(f"{name}\nDir Acc={acc:.3f}")
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")

    # delete unused
    for i in range(len(model_preds), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def direction_summary_table(model_preds: dict, y_true):
    rows = []

    for name, y_pred in model_preds.items():
        acc = direction_accuracy(y_true, y_pred)
        rows.append({
            "model": name,
            "direction_acc": acc
        })

    return pd.DataFrame(rows).sort_values(
        "direction_acc",
        ascending=False
    )

# =====================================
# Multi-model prediction line plot
# =====================================
def plot_model_predictions(
    y_true,
    model_preds: dict,
    title="Model Predictions vs Actual"
):
    plt.figure(figsize=(14,6))
    plt.plot(y_true, label="ACTUAL", linewidth=3)

    for name, y_pred in model_preds.items():
        plt.plot(y_pred, label=name, alpha=0.8)

    plt.title(title)
    plt.xlabel("Time Index (Test Period)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()