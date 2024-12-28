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

# baseline model: logistic regression
log_reg = LogisticRegression(learning_rate=0.01, iterations=1000)
log_reg.fit(X_train, y_train)

y_pred = log_reg.prediction(X_test)

# plot predictions: True Positives, True Negatives, False Positives, False Negatives
# x-axis for true labels, y-axis for predicted labels
y_test_np = np.array(y_test)
y_pred_np = np.array(y_pred)

TP = np.sum((y_test_np == 1) & (y_pred_np == 1))
TN = np.sum((y_test_np == 0) & (y_pred_np == 0))
FP = np.sum((y_test_np == 0) & (y_pred_np == 1))
FN = np.sum((y_test_np == 1) & (y_pred_np == 0))

heatmap_data = np.array([[TN, FP], [FN, TP]])
labels = np.array([["TN", "FP"], ["FN", "TP"]])
annot_labels = np.array([
    [f"True Negatives\n{TN}", f"False Positives\n{FP}"],
    [f"False Negatives\n{FN}", f"True Positives\n{TP}"]
])

plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    heatmap_data, 
    annot=annot_labels, 
    fmt='', 
    cmap="YlOrRd", 
    cbar=True, 
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"]
)

plt.title("Confusion Quadrant Heatmap")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()