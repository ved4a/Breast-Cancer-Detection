from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

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

# view first 5 rows
print(X.head())
print(y.head())

# check shape
print(X.shape)
print(y.shape)

# describe X
print(X.describe())