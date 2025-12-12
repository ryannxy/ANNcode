# -*- coding: utf-8 -*-

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.neural_network import MLPClassifier

# 2. LOAD DATA
try:
    dataset = pd.read_csv('Churn_Modelling.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Churn_Modelling.csv' not found. Please upload the file.")

# 3. DATA PREPROCESSING

# A. Feature Selection
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# B. Encoding Categorical Data
# Gender
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# Geography one-hot encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], 
                       remainder='passthrough')
X = np.array(ct.fit_transform(X))

# 4. DATA SPLITTING (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# 5. FEATURE SCALING
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 6. ANN MODEL (WITHOUT TENSORFLOW)
model = MLPClassifier(
    hidden_layer_sizes=(6, 6),  # 2 hidden layers, 6 neurons each (same as your TF model)
    activation='relu',
    solver='adam',
    alpha=0.0001,               # L2 regularization
    batch_size=32,
    max_iter=200,
    random_state=0,
    verbose=True
)

print("\nTraining ANN model...")
model.fit(X_train, y_train)

# 7. EVALUATION
y_pred = model.predict(X_test)
y_pred_probs = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred_probs)

print("\n--------------------------------------")
print(f"Final Test Accuracy: {acc*100:.2f}%")
print(f"Recall (Sensitivity): {rec*100:.2f}%")
print(f"ROC-AUC Score:       {roc:.4f}")
print("--------------------------------------")

# 8. VISUALIZATIONS
plt.figure(figsize=(18, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
plt.subplot(1, 3, 2)
plt.plot(fpr, tpr, label=f"AUC = {roc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

# Loss Curve
plt.subplot(1, 3, 3)
plt.plot(model.loss_curve_)
plt.title("Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)

plt.tight_layout()
plt.show()
