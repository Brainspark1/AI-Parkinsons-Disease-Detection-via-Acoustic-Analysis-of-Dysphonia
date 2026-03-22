import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ParkinsonNet
from dataset import ParkinsonDataset

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import os
import joblib

# 1. Setup Hyperparameters
INPUT_SIZE = 22  
HIDDEN_SIZE = 64 
OUTPUT_SIZE = 1  
EPOCHS = 50       
BATCH_SIZE = 16   
LEARNING_RATE = 0.001

# 2. Load and Prepare Data
df = pd.read_csv("data/parkinsons.data")
df["subject_id"] = df["name"].apply(lambda x: x.split("_")[2])

X = df.drop(columns=["name", "status", "subject_id"])
y = df["status"]
groups = df["subject_id"]

gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

all_metrics = {
    "accuracy": [],
    "recall": [],
    "precision": [],
    "f1": []
}

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler if you want
    # joblib.dump(scaler, f"models/scaler_fold{fold}.pkl")

    # Dataset and DataLoader
    train_dataset = ParkinsonDataset(X_train_scaled, y_train.values)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    model = ParkinsonNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop (same as before)
    model.train()
    for epoch in range(EPOCHS):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int().numpy().flatten()

    # Store metrics
    all_metrics["accuracy"].append(accuracy_score(y_test.values, preds))
    all_metrics["recall"].append(recall_score(y_test.values, preds))
    all_metrics["precision"].append(precision_score(y_test.values, preds))
    all_metrics["f1"].append(f1_score(y_test.values, preds))

    print(f"Fold {fold} done.")

import numpy as np

print("\n=== 5-Fold Cross-Validation Results ===")
for metric in all_metrics:
    values = all_metrics[metric]
    print(f"{metric.capitalize()}: {np.mean(values):.3f} ± {np.std(values):.3f}")

# 4. Scale (TRAIN ONLY)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler for later use
if not os.path.exists('models'):
    os.makedirs('models')
joblib.dump(scaler, "models/scaler.pkl")

# 5. Create Dataset + Loader
train_dataset = ParkinsonDataset(X_train, y_train.values)
dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 6. Initialize Model
model = ParkinsonNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 7. Training Loop
print(f"Starting Training on {len(train_dataset)} samples...")
model.train()

for epoch in range(EPOCHS):
    running_loss = 0.0

    for data, target in dataloader:
        # Forward
        output = model(data)
        loss = criterion(output, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Avg Loss: {avg_loss:.4f}")

# 8. Evaluation
model.eval()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

with torch.no_grad():
    outputs = model(X_test_tensor)
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).int().numpy().flatten()

y_true = y_test.values

print("\n--- Test Results ---")
print("Accuracy:", accuracy_score(y_true, preds))
print("Recall:", recall_score(y_true, preds))
print("Precision:", precision_score(y_true, preds))
print("F1 Score:", f1_score(y_true, preds))

# 9. Save Model
torch.save(model.state_dict(), "models/parkinsons_model.pth")
print("Model saved to models/parkinsons_model.pth")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test.values, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title(f"Fold {fold} Confusion Matrix")
plt.savefig(f"mlp_confusion_matrix_fold{fold}.png")
plt.close()

torch.save(model.state_dict(), f"models/parkinsons_model_fold{fold}.pth") # save best model