import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions

df = pd.read_csv(r'D:\ML\CODES\Selection\placement.csv')
df = df.iloc[:,1:]

plt.scatter(df['cgpa'],df['iq'], c = df['placement'])
plt.xlabel('cgpa')
plt.ylabel('iq')

X = df.iloc[:,0:2]
Y = df.iloc[:,2:]


X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=0.2)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

model = LogisticRegression()
print(model.fit(X_train,Y_train))

Y_predict = model.predict(X_test)
print(accuracy_score(Y_test,Y_predict))
plot_decision_regions(X_train, Y_train.values.ravel(), clf=model, legend=2)




plt.show()


