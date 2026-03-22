import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ParkinsonNet
from dataset import ParkinsonDataset

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, precision_recall_curve, auc,
    roc_curve, roc_auc_score
)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("data/parkinsons.data")
df["subject_id"] = df["name"].apply(lambda x: x.split("_")[2])

X = df.drop(columns=["name", "status", "subject_id"])
y = df["status"]
groups = df["subject_id"]

gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# -----------------------------
# 2. Define models
# -----------------------------
sklearn_models = {
    "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

mlp_params = {
    "input_size": X.shape[1],
    "hidden_size": 64,
    "output_size": 1,
    "epochs": 50,
    "batch_size": 16,
    "lr": 0.001
}

all_results = {}

# -----------------------------
# 3. Evaluate sklearn models
# -----------------------------
for model_name, model in sklearn_models.items():
    print(f"\n=== Evaluating {model_name} ===")

    metrics = {"accuracy": [], "recall": [], "precision": [], "f1": [], "specificity": [], "pr_auc": []}
    
    all_probs = []
    all_true = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_test_scaled)[:,1]
        preds = (probs > 0.5).astype(int)

        # Store for ROC
        all_probs.extend(probs)
        all_true.extend(y_test)

        # Metrics
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)
        prec = precision_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        spec = tn / (tn + fp)

        precision_vals, recall_vals, _ = precision_recall_curve(y_test, probs)
        pr_auc_val = auc(recall_vals, precision_vals)

        metrics["accuracy"].append(acc)
        metrics["recall"].append(rec)
        metrics["precision"].append(prec)
        metrics["f1"].append(f1)
        metrics["specificity"].append(spec)
        metrics["pr_auc"].append(pr_auc_val)

        # Confusion matrix (fold 1)
        if fold == 1:
            cm = confusion_matrix(y_test, preds)
            plt.figure()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f"{model_name} Confusion Matrix")
            plt.colorbar()
            plt.xticks([0,1], ['Healthy','Parkinson'])
            plt.yticks([0,1], ['Healthy','Parkinson'])
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
            plt.tight_layout()
            plt.savefig(f"{model_name.replace(' ','_')}_confusion_matrix.png")
            plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(all_true, all_probs)
    roc_auc = roc_auc_score(all_true, all_probs)

    all_results[model_name] = {
        **{metric: f"{np.mean(vals):.3f} ± {np.std(vals):.3f}" for metric, vals in metrics.items()},
        "roc_curve": (fpr, tpr, roc_auc)
    }

# -----------------------------
# 4. Evaluate MLP
# -----------------------------
print("\n=== Evaluating MLP (Proposed) ===")

mlp_metrics = {"accuracy": [], "recall": [], "precision": [], "f1": [], "specificity": [], "pr_auc": []}
mlp_all_probs = []
mlp_all_true = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = ParkinsonDataset(X_train_scaled, y_train.values)
    train_loader = DataLoader(train_dataset, batch_size=mlp_params["batch_size"], shuffle=True)

    model = ParkinsonNet(mlp_params["input_size"], mlp_params["hidden_size"], mlp_params["output_size"])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=mlp_params["lr"])

    model.train()
    for epoch in range(mlp_params["epochs"]):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    model.eval()
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        probs = torch.sigmoid(outputs).numpy().flatten()
        preds = (probs > 0.5).astype(int)

    # Store for ROC
    mlp_all_probs.extend(probs)
    mlp_all_true.extend(y_test.values)

    acc = accuracy_score(y_test.values, preds)
    rec = recall_score(y_test.values, preds)
    prec = precision_score(y_test.values, preds)
    f1 = f1_score(y_test.values, preds)
    tn, fp, fn, tp = confusion_matrix(y_test.values, preds).ravel()
    spec = tn / (tn + fp)

    precision_vals, recall_vals, _ = precision_recall_curve(y_test.values, probs)
    pr_auc_val = auc(recall_vals, precision_vals)

    mlp_metrics["accuracy"].append(acc)
    mlp_metrics["recall"].append(rec)
    mlp_metrics["precision"].append(prec)
    mlp_metrics["f1"].append(f1)
    mlp_metrics["specificity"].append(spec)
    mlp_metrics["pr_auc"].append(pr_auc_val)

# ROC for MLP
fpr, tpr, _ = roc_curve(mlp_all_true, mlp_all_probs)
roc_auc = roc_auc_score(mlp_all_true, mlp_all_probs)

all_results["MLP (Proposed)"] = {
    **{metric: f"{np.mean(vals):.3f} ± {np.std(vals):.3f}" for metric, vals in mlp_metrics.items()},
    "roc_curve": (fpr, tpr, roc_auc)
}

# -----------------------------
# 5. Plot ROC curves
# -----------------------------
plt.figure(figsize=(8,6))

for model_name, result in all_results.items():
    fpr, tpr, roc_auc = result["roc_curve"]
    plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC={roc_auc:.3f})")

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison for All Models")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve_all_models.png", dpi=300)
plt.show()

# -----------------------------
# 6. Save metrics
# -----------------------------
metrics_df = pd.DataFrame({k: {m:v for m,v in res.items() if m != "roc_curve"} 
                           for k,res in all_results.items()}).T

metrics_df.to_csv("all_model_metrics.csv")
metrics_df.to_json("all_model_metrics.json", orient="index")

print("Saved ROC curve + metrics")

# -----------------------------
# 7. Bar Graph Comparison
# -----------------------------
metrics_to_plot = ["accuracy", "recall", "precision", "f1", "specificity", "pr_auc"]

models = list(all_results.keys())

# Extract mean values (ignore ± std)
data = {metric: [] for metric in metrics_to_plot}

for model in models:
    for metric in metrics_to_plot:
        val = all_results[model][metric]
        mean_val = float(val.split("±")[0].strip())
        data[metric].append(mean_val)

# Plot
x = np.arange(len(models))
width = 0.12

plt.figure(figsize=(12,6))

for i, metric in enumerate(metrics_to_plot):
    plt.bar(x + i*width, data[metric], width, label=metric.capitalize())

plt.xticks(x + width*2.5, models, rotation=30)
plt.ylabel("Score")
plt.title("Comparison of Models Across Evaluation Metrics")
plt.legend()
plt.tight_layout()

plt.savefig("model_metrics_bar_chart.png", dpi=300)
print("Metrics chart saved")
plt.show()