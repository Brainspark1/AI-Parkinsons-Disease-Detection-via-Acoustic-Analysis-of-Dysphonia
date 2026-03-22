import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ParkinsonNet
from dataset import ParkinsonDataset
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_curve, auc
import numpy as np
import os
import matplotlib.pyplot as plt
import json

# -----------------------------
# 1. Hyperparameters
# -----------------------------
INPUT_SIZE = 22
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# -----------------------------
# 2. Load data
# -----------------------------
df = pd.read_csv("data/parkinsons.data")
df["subject_id"] = df["name"].apply(lambda x: x.split("_")[2])

X = df.drop(columns=["name", "status", "subject_id"])
y = df["status"]
groups = df["subject_id"]

# -----------------------------
# 3. Stratified Group K-Fold
# -----------------------------
gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Create models folder if not exists
if not os.path.exists('models'):
    os.makedirs('models')

# -----------------------------
# 4. Function to compute metrics
# -----------------------------
def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return accuracy, recall, precision, f1, specificity

# -----------------------------
# 5. Models dictionary to store metrics
# -----------------------------
model_names = ["MLP (Proposed)"]  # Add more models here if you implement SVM, RF, etc.
model_metrics = {name: {"accuracy": [], "recall": [], "precision": [], "f1": [], "specificity": []} for name in model_names}

# -----------------------------
# 6. Loop over folds for MLP (example)
# -----------------------------
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
    print(f"\n=== Fold {fold} ===")
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Scale only on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Dataset and DataLoader
    train_dataset = ParkinsonDataset(X_train_scaled, y_train.values)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    model = ParkinsonNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Avg Loss: {running_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probs = torch.sigmoid(outputs).numpy().flatten()
        preds = (probs > 0.5).astype(int)

    # Compute metrics
    acc, rec, prec, f1, spec = compute_metrics(y_test.values, preds)
    print(f"Fold {fold} - Acc: {acc:.3f}, Rec: {rec:.3f}, Spec: {spec:.3f}")

    # Store metrics
    for metric_name, value in zip(["accuracy","recall","precision","f1","specificity"], [acc, rec, prec, f1, spec]):
        model_metrics["MLP (Proposed)"][metric_name].append(value)

    # Save fold model
    torch.save(model.state_dict(), f"models/mlp_fold{fold}.pth")

# -----------------------------
# 7. Print and save averaged metrics
# -----------------------------
print("\n=== 5-Fold CV Results ===")
final_results = []
for model_name, metrics in model_metrics.items():
    result_row = {"Model": model_name}
    for metric, values in metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{model_name} - {metric.capitalize()}: {mean_val:.3f} ± {std_val:.3f}")
        result_row[metric.capitalize()] = f"{mean_val:.3f} ± {std_val:.3f}"
    final_results.append(result_row)

df_results = pd.DataFrame(final_results)
df_results.to_csv("all_model_metrics_with_specificity.csv", index=False)
print("\nSaved metrics to 'all_model_metrics_with_specificity.csv'")