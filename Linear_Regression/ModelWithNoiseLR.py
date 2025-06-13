import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.array([1,2,3,4,5,6,7,8,9,10,11]).reshape(-1, 1)
Y = np.array([10,20,30,40,50,60,70,80,90,120,140]).reshape(-1, 1)  # Notice: last value is 150 (should be 100)


data = LinearRegression()
data.fit(X,Y)

print(data.score(X,Y))

print("Prediction with noise at X=11:", data.predict([[12]]))
plt.scatter(X, Y, color='blue', label='Data with noise')
plt.plot(X, data.predict(X), color='red', label='Model with noise')
plt.title("Linear Regression with Outlier")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
