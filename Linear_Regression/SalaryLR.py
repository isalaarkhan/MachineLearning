import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

Years = np.array([0,2,4,5,8,10]).reshape(-1, 1)  # Square feet
Salary = np.array([100,300,500,700,900,1100]).reshape(-1, 1) #Price

year_train,year_test, salary_train , salary_test = train_test_split(Years,Salary,test_size= 0.2)

model = LinearRegression()
model.fit(year_train,salary_train)


print(model.score(year_test,salary_test))


plt.scatter(year_train,salary_train)
plt.xlabel("Years")
plt.ylabel("Salary")
plt.plot(Years, model.predict(Years))
plt.title("Experience and Salary")
plt.show()