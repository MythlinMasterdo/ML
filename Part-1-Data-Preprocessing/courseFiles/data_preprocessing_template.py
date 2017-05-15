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
                
#Taking care of missing data
from sklearn.preprocessing import Imputer
#Do command I for var info about Imputer class
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
#selecting indices 1,2 here, lowerbound is inclusive, higherbound is not
imputer = imputer.fit(X[:, 1:3])
#setting the value of X at these indices to the new dataset values
X[:, 1:3] = imputer.transform(X[:, 1:3])


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_X.fit_transform(y)

#splitting dataset
from sklearn.model_selection import train_test_split
#Random_state was only used to get some results as instructor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
#train set must be fit AND transformed
X_train = sc_X.fit_transform(X_train)
#test must just be transformed
X_test = sc_X.transform(X_test)
