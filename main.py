import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- STEP 1: LOAD DATA ---
data = load_breast_cancer()[cite: 1]
X, y = data.data, data.target[cite: 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)[cite: 1]

# --- STEP 2: UNSUPERVISED LEARNING (Materi Sebelumnya) ---
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)[cite: 1]
# Evaluasi Clustering dengan Confusion Matrix (Ref: ksnugroho)
cm_cluster = confusion_matrix(y, clusters)

# --- STEP 3: ENSEMBLE LEARNING (Materi DM06.pdf) ---
# 1. Bagging (Random Forest)
rf = RandomForestClassifier(n_estimators=100, random_state=42)[cite: 1]
rf.fit(X_train, y_train)

# 2. Boosting (Gradient Boosting)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)[cite: 1]
gb.fit(X_train, y_train)

# 3. Stacking
base_models = [('rf', rf), ('gb', gb)]
stacking = StackingClassifier(
    estimators=base_models, 
    final_estimator=LogisticRegression(),
    cv=5
)[cite: 1]
stacking.fit(X_train, y_train)

# --- STEP 4: EVALUASI & VISUALISASI ---
models = {'Random Forest': rf, 'Gradient Boosting': gb, 'Stacking': stacking}
for name, model in models.items():
    acc = accuracy_score(y_test, model.predict(X_test))[cite: 1]
    print(f"Akurasi {name}: {acc:.4f}")

# Simpan Grafik Confusion Matrix untuk Website
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, stacking.predict(X_test)), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Stacking Model')
plt.savefig('confusion_matrix.png') # File ini akan dipanggil di HTML
