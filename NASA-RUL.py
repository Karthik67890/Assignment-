import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Here we  Load Our dataset (we replace 'data.csv' with our dataset file)
data = pd.read_csv('data.csv')
 
# We assume our dataset has columns like feature 1, feature 2, feature 3, and 'RUL'
X = data[['feature1', 'feature2', 'feature3']]
y = data['RUL']

# Here we split the data into 2 sets, i.e, training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  We are creating a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# This is the evaluvation part of the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared (R2) Score:", r2)

# Predict RUL for new data
new_data = pd.DataFrame({'feature1': [value1], 'feature2': [value2], 'feature3': [value3]})
predicted_rul = model.predict(new_data)
print("Predicted RUL:", predicted_rul[0])
