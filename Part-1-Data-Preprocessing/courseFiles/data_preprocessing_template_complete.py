# Data Preprocessing Template

# Importing the libraries
import numpy as np
#did this to see entire array
np.set_printoptions(threshold = np.nan)
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
#train set must be fit AND transformed
X_train = sc_X.fit_transform(X_train)
#test must just be transformed
X_test = sc_X.transform(X_test)"""

