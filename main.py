import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, 
                              AdaBoostClassifier, GradientBoostingClassifier, 
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Dataset (Menggunakan Breast Cancer sesuai materi DM06.pdf)
data = load_breast_cancer()[cite: 1]
X, y = data.data, data.target[cite: 1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42[cite: 1]
)

# 2. Inisialisasi Model Tunggal (Base Model)
dt = DecisionTreeClassifier(random_state=42)[cite: 1]

# 3. Implementasi Bagging & Random Forest
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)[cite: 1]

rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt', # sqrt(p) untuk klasifikasi
    random_state=42,
    n_jobs=-1
)[cite: 1]

# 4. Implementasi Boosting (AdaBoost & Gradient Boosting)
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1), # Decision stump
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)[cite: 1]

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)[cite: 1]

# 5. Implementasi Stacking
base_models = [
    ('rf', rf),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),[cite: 1]
    ('knn', KNeighborsClassifier(n_neighbors=5)),[cite: 1]
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42))[cite: 1]
]

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000), # Meta-learner
    cv=5,
    stack_method='predict_proba'
)[cite: 1]

# 6. Pelatihan dan Evaluasi
models = {
    'Decision Tree': dt,
    'Bagging': bagging,
    'Random Forest': rf,
    'AdaBoost': ada,
    'Gradient Boosting': gb,
    'Stacking': stacking
}

results = {}

print("--- EVALUASI PERFORMA MODEL ---")
for name, model in models.items():
    model.fit(X_train, y_train)[cite: 1]
    predictions = model.predict(X_test)[cite: 1]
    acc = accuracy_score(y_test, predictions)[cite: 1]
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# 7. Visualisasi Confusion Matrix (Sesuai artikel Medium yang Anda tanyakan)
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, models['Random Forest'].predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 8. Feature Importance (Random Forest)
importance_df = pd.DataFrame({
    'feature': data.feature_names,
    'importance': rf.feature_importances_[cite: 1]
}).sort_values('importance', ascending=False)

print("\nTop 5 Fitur Terpenting (Random Forest):")
print(importance_df.head())[cite: 1]
