# Activatino Function:
#   - Binary Step, ReLU, Sigmoid/Logistic Curve, Linear, Softmax, Rectified, Exponential Linear Unit (ELU)
# Feed Forward:
#   - 
# Back Propagation: Weights are updated to minimize the calculated error
#   - 
#   * Gradient Descent
#   * Cost Function: (sum of square differences b/w the output and the expected result)

# from sklearn.model_selection import train_test_split
import os # operating system
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier # multi layer perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the Iris dataset (NOTE: the data is in iris.data (data attribute) dimension 150x4)
iris = load_iris()

print(f'measurements/features: {iris.feature_names}') # name of the 4 features
print(f'Species: {iris.target_names}\n') # 0 = Setosa, 1 = Vesicolor, 2 = Virginica 
# print(f'Species representation: {iris.target}\n') # intigers representig the 3 species

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Standardization
scaler_standard = StandardScaler()
df_standardized = pd.DataFrame(scaler_standard.fit_transform(df.iloc[:, :-1]), columns=df.columns[:-1])
df_standardized['species'] = df['species']


# Normalization
scaler_normalize = MinMaxScaler()
df_normalized = pd.DataFrame(scaler_normalize.fit_transform(df.iloc[:, :-1]), columns=df.columns[:-1])
df_normalized['species'] = df['species']


# Separate features and target
X = df_normalized.iloc[:, :-1]
y = df_normalized['species'].cat.codes

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', max_iter=1000, random_state=42)

# Train the model
mlp.fit(X_train, y_train)

# Evaluate the model
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.2f}')

# Calculate the training accuracy
train_accuracy = mlp.score(X_train, y_train)
print(f'Training Accuracy: {train_accuracy:.2f}')

# Print classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Print confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

#------------------------------------------------------
#------------------------------------------------------
# DATASET
# Loading the dataset
# iris = load_iris()
# Assinging the Data (X) and the Labels (y)
# print(iris.data)
# X = iris.data # Matrix
# y = iris.target # vector
