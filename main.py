from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from logistic_reg import LogisticRegression

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
# y_test_np = np.array(y_test)

# plt.figure(figsize=(10, 6))

# plt.scatter(np.where((y_test_np == 1) & (y_pred == 1))[0], 
#             np.ones(len(np.where((y_test_np == 1) & (y_pred == 1))[0])), 
#             color='green', label='True Positives (TP)')

# plt.scatter(np.where((y_test_np == 0) & (y_pred == 0))[0], 
#             np.zeros(len(np.where((y_test_np == 0) & (y_pred == 0))[0])), 
#             color='blue', label='True Negatives (TN)')

# plt.scatter(np.where((y_test_np == 0) & (y_pred == 1))[0], 
#             np.ones(len(np.where((y_test_np == 0) & (y_pred == 1))[0])), 
#             color='red', label='False Positives (FP)')

# plt.scatter(np.where((y_test_np == 1) & (y_pred == 0))[0], 
#             np.zeros(len(np.where((y_test_np == 1) & (y_pred == 0))[0])), 
#             color='yellow', label='False Negatives (FN)')

# # quadrants
# plt.axhline(0.5, color='black', linestyle='--', linewidth=1)
# plt.axvline(len(y_test) / 2, color='black', linestyle='--', linewidth=1)

# plt.title("Logistic Regression Predictions")
# plt.xlabel("Sample Index")
# plt.yticks([0, 1], ["Benign (0)", "Malignant (1)"])
# plt.legend()
# plt.grid(alpha=0.3)

# # Display the plot
# plt.show()
