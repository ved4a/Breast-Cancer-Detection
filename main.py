from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from logistic_reg import LogisticRegression
from neural_network import NeuralNetwork

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# import data
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets['Diagnosis']

# map 'malignant' and 'benign' to 1 and 0
y = y.map({'M': 1, 'B' : 0})

# split into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale data for better performance
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression [Baseline Model]
log_reg = LogisticRegression(learning_rate=0.01, iterations=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.prediction(X_test)

# Logistic Regression: Confusion Matrix and Heatmap
y_test_np = np.array(y_test)
y_pred_log_reg_np = np.array(y_pred_log_reg)

TP_log_reg = np.sum((y_test_np == 1) & (y_pred_log_reg_np == 1))
TN_log_reg = np.sum((y_test_np == 0) & (y_pred_log_reg_np == 0))
FP_log_reg = np.sum((y_test_np == 0) & (y_pred_log_reg_np == 1))
FN_log_reg = np.sum((y_test_np == 1) & (y_pred_log_reg_np == 0))

heatmap_data_log_reg = np.array([[TN_log_reg, FP_log_reg], [FN_log_reg, TP_log_reg]])
annot_labels_log_reg = np.array([
    [f"True Negatives\n{TN_log_reg}", f"False Positives\n{FP_log_reg}"],
    [f"False Negatives\n{FN_log_reg}", f"True Positives\n{TP_log_reg}"]
])

plt.figure(figsize=(8, 6))
sns.heatmap(
    heatmap_data_log_reg,
    annot=annot_labels_log_reg,
    fmt='',
    cmap="YlOrRd",
    cbar=True,
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"]
)
plt.title("Logistic Regression: Confusion Quadrant")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()