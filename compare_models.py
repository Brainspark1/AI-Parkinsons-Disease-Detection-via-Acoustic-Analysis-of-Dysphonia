import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

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
    X = df.drop(['name', 'status'], axis=1)
    y = df['status']
except FileNotFoundError:
    print("Error: data/parkinsons.data not found. Please ensure the file is in the data folder.")
    exit()

# 2. Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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

for name, model in models.items():
    cv_results = cross_validate(model, X_scaled, y, cv=5, scoring=metrics)
    
    acc = cv_results['test_accuracy'].mean()
    rec = cv_results['test_recall'].mean()
    prec = cv_results['test_precision'].mean()
    f1 = cv_results['test_f1'].mean()
    
    performance_summary.append({'Name': name, 'Accuracy': acc, 'Recall': rec, 'F1': f1})
    print(f"{name:<20} | {acc:.4f}     | {rec:.4f}     | {f1:.4f}")

# 5. Optional: Quick Visualization
summary_df = pd.DataFrame(performance_summary).sort_values(by='Recall', ascending=False)
summary_df.plot(x='Name', y=['Accuracy', 'Recall'], kind='bar', figsize=(10, 6))
plt.title("Model Comparison (Sorted by Recall)")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()