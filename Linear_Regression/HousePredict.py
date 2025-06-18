import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error ,r2_score

df = pd.read_csv('house_price_data.csv')

X = df.drop('Price', axis= 1)
Y = df['Price']

X_test , X_train, Y_test , Y_train = train_test_split(X,Y,test_size=0.8)

scl = StandardScaler()

X_train_scl = scl.fit_transform(X_train)
X_test_scl = scl.fit_transform(X_test)
Y_train_scl = scl.fit_transform(Y_train.values.reshape(-1, 1))
Y_test_scl = scl.fit_transform(Y_test.values.reshape(-1, 1))

model = LinearRegression()
model.fit(X_train_scl,Y_train_scl)

y_pred = model.predict(X_test_scl)

mse = mean_squared_error(Y_test_scl,y_pred)
r2 = r2_score(Y_test_scl,y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)