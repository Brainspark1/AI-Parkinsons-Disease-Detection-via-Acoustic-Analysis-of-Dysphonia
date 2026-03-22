import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import os

# ---------------------------
# 1️⃣ PyTorch MLP Dataset
# ---------------------------
class ParkinsonDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------
# 2️⃣ MLP Model
# ---------------------------
class ParkinsonNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParkinsonNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # raw logits for BCEWithLogitsLoss

# ---------------------------
# 3️⃣ Load data
# ---------------------------
df = pd.read_csv("data/parkinsons.data")
df["subject_id"] = df["name"].apply(lambda x: x.split("_")[2])

X = df.drop(columns=["name", "status", "subject_id"])
y = df["status"]
groups = df["subject_id"]

# ---------------------------
# 4️⃣ Classical ML Models
# ---------------------------
sklearn_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF)": SVC(kernel='rbf', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}

# ---------------------------
# 5️⃣ 5-Fold Stratified Group CV
# ---------------------------
gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Storage for metrics
final_results = {}

# ---------------------------
# 6️⃣ Loop over folds
# ---------------------------
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
    print(f"\n=== Fold {fold} ===")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------
    # 6a️⃣ Train MLP
    # ---------------------------
    INPUT_SIZE = X_train_scaled.shape[1]
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 1
    EPOCHS = 50
    BATCH_SIZE = 16
    LR = 0.001

    train_dataset = ParkinsonDataset(X_train_scaled, y_train.values)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ParkinsonNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

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
    metrics = {
        "accuracy": accuracy_score(y_test.values, preds),
        "recall": recall_score(y_test.values, preds),
        "precision": precision_score(y_test.values, preds),
        "f1": f1_score(y_test.values, preds)
    }
    if "MLP (Proposed)" not in final_results:
        final_results["MLP (Proposed)"] = {"accuracy": [], "recall": [], "precision": [], "f1": []}
    for k in metrics:
        final_results["MLP (Proposed)"][k].append(metrics[k])

    # ---------------------------
    # 6b️⃣ Train Classical ML Models
    # ---------------------------
    for name, clf in sklearn_models.items():
        clf.fit(X_train_scaled, y_train)
        preds_clf = clf.predict(X_test_scaled)
        if name not in final_results:
            final_results[name] = {"accuracy": [], "recall": [], "precision": [], "f1": []}
        final_results[name]["accuracy"].append(accuracy_score(y_test, preds_clf))
        final_results[name]["recall"].append(recall_score(y_test, preds_clf))
        final_results[name]["precision"].append(precision_score(y_test, preds_clf))
        final_results[name]["f1"].append(f1_score(y_test, preds_clf))

# ---------------------------
# 7️⃣ Compute mean ± std and create table
# ---------------------------
summary = []
for name, metrics in final_results.items():
    summary.append({
        "Model": name,
        "Accuracy": f"{np.mean(metrics['accuracy']):.3f} ± {np.std(metrics['accuracy']):.3f}",
        "Recall": f"{np.mean(metrics['recall']):.3f} ± {np.std(metrics['recall']):.3f}",
        "Precision": f"{np.mean(metrics['precision']):.3f} ± {np.std(metrics['precision']):.3f}",
        "F1": f"{np.mean(metrics['f1']):.3f} ± {np.std(metrics['f1']):.3f}"
    })

summary_df = pd.DataFrame(summary)
summary_df = summary_df.sort_values(by="F1", ascending=False)
summary_df.to_csv("final_comparative_table.csv", index=False)
print("\n=== Final Comparative Table ===")
print(summary_df)