import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

# 1. Load data
df = pd.read_csv("data/parkinsons.data")
df["subject_id"] = df["name"].apply(lambda x: x.split("_")[2])

X = df.drop(columns=["name", "status", "subject_id"])
y = df["status"]
groups = df["subject_id"]

# 2. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF)": SVC(kernel='rbf', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}

# 3. Stratified Group K-Fold
gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Storage
results = {name: {"accuracy": [], "recall": [], "precision": [], "f1": []} for name in models}

# 5. Loop over folds
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        results[name]["accuracy"].append(accuracy_score(y_test, preds))
        results[name]["recall"].append(recall_score(y_test, preds))
        results[name]["precision"].append(precision_score(y_test, preds))
        results[name]["f1"].append(f1_score(y_test, preds))

# 6. Summary table
summary = []
for name, metrics in results.items():
    summary.append({
        "Model": name,
        "Accuracy": f"{np.mean(metrics['accuracy']):.3f} ± {np.std(metrics['accuracy']):.3f}",
        "Recall": f"{np.mean(metrics['recall']):.3f} ± {np.std(metrics['recall']):.3f}",
        "Precision": f"{np.mean(metrics['precision']):.3f} ± {np.std(metrics['precision']):.3f}",
        "F1": f"{np.mean(metrics['f1']):.3f} ± {np.std(metrics['f1']):.3f}"
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("sklearn_results.csv", index=False)
print(summary_df)