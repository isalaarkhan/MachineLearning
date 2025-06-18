import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Advertising_dataset.csv')
#print(df.head())

#sns.pairplot(df, x_vars=['TV','Radio','Newspaper'],y_vars=['Sales'], kind= 'scatter')
#plt.show()

X = df.iloc[:,1:4]
Y = df['Sales']

X_train ,X_test, Y_train ,Y_test = train_test_split(X,Y,test_size=0.2)

model = LinearRegression()
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

print("Mean Squared Error:", mean_squared_error(Y_test, y_pred))
print("R^2 Score:", r2_score(Y_test, y_pred))

plt.scatter(Y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red')  # Perfect prediction line
plt.show()
