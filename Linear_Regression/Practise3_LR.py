import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

time_studied = np.array([20,50,32,65,23,43,10,5,22,35,29,5,56]).reshape(-1,1)
scores = np.array([56,83,47,93,47,82,45,23,55,67,57,4,89]).reshape(-1,1)

test_size = np.linspace(0.1,0,9,9)
result=[]

for size in test_size:
    X_train,X_test,Y_train,Y_test = train_test_split(time_studied,scores,test_size = size, random_state=42)

    model = LinearRegression()
    model.fit(X_train,Y_train)

    score = model.score(X_test,Y_test)
    result.append(score)
    print(f"Test size:{size :.1%} -> R^2 : { score :.4f}")

plt.plot(test_size,result)
plt.title("Test size Vs Model Acuraccy")
plt.Xlable("Test Set Proportion")
plt.ylabel("R^2 Score")
plt.grid(True)
plt.show()