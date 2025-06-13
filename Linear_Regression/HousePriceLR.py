import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.array([800, 1000, 1200, 1500, 1800]).reshape(-1, 1)  # Square feet
Y = np.array([100000, 150000, 180000, 200000, 250000]).reshape(-1, 1) #Price


model = LinearRegression()
model.fit(X,Y)


print(model.predict(np.array([1300]).reshape(-1,1)))


plt.scatter(X,Y)
plt.xlabel("Size")
plt.ylabel("Price")
plt.plot(X, model.predict(X))
plt.title("House Price vs Size")
plt.show()