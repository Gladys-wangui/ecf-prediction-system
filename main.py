import tkinter as tk 
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

#increase maximum iterations
model = LogisticRegression(max_iter=1000)

# Assuming your dataset is saved as 'ecf.csv'
dataset_path = 'ecf.csv'

# Load the dataset
df = pd.read_csv(dataset_path)

# Separate features (X) and labels (y)
X = df.drop('East_Coast_Fever', axis=1)
y = df['East_Coast_Fever']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')


# Save the trained model
model_filename = 'ecf_model.joblib'
joblib.dump(model, 'trained_model.joblib')
print(f'Trained model saved as {model_filename}')

