import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# 1. Load Data
try:
    df = pd.read_csv('data/parkinsons.data')
    # Dropping 'name' as it's a string ID, 'status' is our target
    X = df.drop(['name', 'status', 'subject_id'], axis=1)
    y = df['status']
    groups = df['subject_id']
except FileNotFoundError:
    print("Error: data/parkinsons.data not found. Please ensure the file is in the data folder.")
    exit()

df["subject_id"] = df["name"].apply(lambda x: x.split("_")[2])

# 3. Define the Model "Zoo"
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)": SVC(kernel='rbf', probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}

# 4. Evaluation Metrics
metrics = ['accuracy', 'recall', 'precision', 'f1']

print(f"{'Model':<20} | {'Accuracy':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 65)

performance_summary = []

gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

print(f"{'Model':<20} | {'Accuracy':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 65)

performance_summary = []

for name, model in models.items():
    accuracies, recalls, precisions, f1s = [], [], [], []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))

    acc = np.mean(accuracies)
    rec = np.mean(recalls)
    f1 = np.mean(f1s)

    performance_summary.append({'Name': name, 'Accuracy': acc, 'Recall': rec, 'F1': f1})
    print(f"{name:<20} | {acc:.4f}     | {rec:.4f}     | {f1:.4f}")