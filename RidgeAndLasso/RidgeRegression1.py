import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv(r'D:\ML\CODES\Basics\house_price_data.csv')

# Features and target
X = df.drop('Price', axis=1)
Y = df['Price']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8)

# Scaling
scl_X = StandardScaler()
X_train_scl = scl_X.fit_transform(X_train)
X_test_scl = scl_X.transform(X_test)

scl_Y = StandardScaler()
Y_train_scl = scl_Y.fit_transform(Y_train.values.reshape(-1, 1))
Y_test_scl = scl_Y.transform(Y_test.values.reshape(-1, 1))

# Ridge Regression + GridSearchCV
model = Ridge()
params = {'alpha': [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]}
ridgeCV = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=5)

# Fit model
ridgeCV.fit(X_train_scl, Y_train_scl)

# Predict
y_pred = ridgeCV.predict(X_test_scl)

# Results
print("Best Alpha:", ridgeCV.best_params_)
print("Best Score (Neg. MSE):", ridgeCV.best_score_)
print("MSE:", mean_squared_error(Y_test_scl, y_pred))
print("R² Score:", r2_score(Y_test_scl, y_pred))
