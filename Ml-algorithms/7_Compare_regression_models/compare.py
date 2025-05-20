# Step 1 - Load Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values  # Keep as 2D array
y = dataset.iloc[:, 2].values   # 1D array

###########################
### Linear Regression ###
###########################
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)
lin_pred = linear_regressor.predict([[6.5]])
print('The predicted salary of a person at 6.5 Level with Linear Regression is ', lin_pred)

################################
### Polynomial Regression ###
################################
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=4)
X_poly = poly_features.fit_transform(X)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y)
poly_pred = poly_regressor.predict(poly_features.transform([[6.5]]))
print('The predicted salary of a person at 6.5 Level with Polynomial Regression is ', poly_pred)

################################
### Support Vector Regression ###
################################
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ss_x = StandardScaler()
ss_y = StandardScaler()

X_scaled = ss_x.fit_transform(X)
y_scaled = ss_y.fit_transform(y.reshape(-1, 1))  # Reshape y

svr_regressor = SVR(kernel="rbf")
svr_regressor.fit(X_scaled, y_scaled.ravel())  # ravel() to flatten

position_val = ss_x.transform([[6.5]])
pred_val_scaled = svr_regressor.predict(position_val)
svr_pred = ss_y.inverse_transform(pred_val_scaled.reshape(-1, 1))
print('The predicted salary of a person at 6.5 Level with Support Vector Regression is ', svr_pred)

################################
### Decision Tree Regression ###
################################
from sklearn.tree import DecisionTreeRegressor
tree_regressor = DecisionTreeRegressor(criterion="squared_error", random_state=0)
tree_regressor.fit(X, y)
tree_pred = tree_regressor.predict([[6.5]])
print('The predicted salary of a person at 6.5 Level with Decision Tree Regression is ', tree_pred)

################################
### Random Forest Regression ###
################################
from sklearn.ensemble import RandomForestRegressor
forest_regressor = RandomForestRegressor(n_estimators=300, random_state=0)
forest_regressor.fit(X, y)
forest_pred = forest_regressor.predict([[6.5]])
print('The predicted salary of a person at 6.5 Level with Random Forest Regression is ', forest_pred)

################################
### Visualizations ###
################################
X_grid = np.arange(X.min(), X.max(), 0.01).reshape(-1, 1)

plt.scatter(X, y, color="red")
plt.plot(X_grid, linear_regressor.predict(X_grid), color="blue", label="Linear")
plt.plot(X_grid, poly_regressor.predict(poly_features.transform(X_grid)), color="green", label="Polynomial")

# SVR predictions (properly reshaped)
svr_predictions_scaled = svr_regressor.predict(ss_x.transform(X_grid)).reshape(-1, 1)
svr_predictions = ss_y.inverse_transform(svr_predictions_scaled)
plt.plot(X_grid, svr_predictions, color="orange", label="SVR")

plt.plot(X_grid, tree_regressor.predict(X_grid), color="black", label="Decision Tree")
plt.plot(X_grid, forest_regressor.predict(X_grid), color="purple", label="Random Forest")

plt.title("Regression Models Comparison")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.legend()
plt.tight_layout()
plt.show()
