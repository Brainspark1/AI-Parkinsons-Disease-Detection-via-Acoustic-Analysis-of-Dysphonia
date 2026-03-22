import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ParkinsonNet
from dataset import ParkinsonDataset

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("data/parkinsons.data")
df["subject_id"] = df["name"].apply(lambda x: x.split("_")[2])

X = df.drop(columns=["name", "status", "subject_id"])
y = df["status"]
groups = df["subject_id"]

# -----------------------------
# 2. Stratified Group K-Fold
# -----------------------------
gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# -----------------------------
# 3. Models to evaluate
# -----------------------------
sklearn_models = {
    "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# Hyperparameters for MLP
INPUT_SIZE = 22
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# Storage for metrics
results = []

# -----------------------------
# 4. Loop over models
# -----------------------------
for model_name, model in list(sklearn_models.items()) + [("MLP (Proposed)", "MLP")]:
    acc_list, rec_list, prec_list, f1_list, spec_list = [], [], [], [], []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale features for all models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # -----------------------------
        # Train sklearn models
        # -----------------------------
        if model_name != "MLP (Proposed)":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            # -----------------------------
            # Train MLP
            # -----------------------------
            train_dataset = ParkinsonDataset(X_train_scaled, y_train.values)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            mlp_model = ParkinsonNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(mlp_model.parameters(), lr=LEARNING_RATE)
            
            mlp_model.train()
            for epoch in range(EPOCHS):
                for data, target in train_loader:
                    optimizer.zero_grad()
                    output = mlp_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            mlp_model.eval()
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            with torch.no_grad():
                outputs = mlp_model(X_test_tensor)
                probs = torch.sigmoid(outputs).numpy().flatten()
                preds = (probs > 0.5).astype(int)
        
        # -----------------------------
        # Compute metrics
        # -----------------------------
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)
        prec = precision_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        spec = tn / (tn + fp)
        
        acc_list.append(acc)
        rec_list.append(rec)
        prec_list.append(prec)
        f1_list.append(f1)
        spec_list.append(spec)
    
    # Average ± std
    results.append({
        "Model": model_name,
        "Accuracy": f"{np.mean(acc_list):.3f} ± {np.std(acc_list):.3f}",
        "Recall": f"{np.mean(rec_list):.3f} ± {np.std(rec_list):.3f}",
        "Precision": f"{np.mean(prec_list):.3f} ± {np.std(prec_list):.3f}",
        "F1": f"{np.mean(f1_list):.3f} ± {np.std(f1_list):.3f}",
        "Specificity": f"{np.mean(spec_list):.3f} ± {np.std(spec_list):.3f}"
    })

# -----------------------------
# 5. Save results
# -----------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("all_models_metrics_with_specificity.csv", index=False)
print("Results saved to all_models_metrics_with_specificity.csv")
print(results_df)