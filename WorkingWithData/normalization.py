import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df= pd.read_csv('wine_data.csv',usecols=[0,1,2])
df.columns =['Class label','Alcohol','Malic acid']


X_train, X_test, Y_train,Y_test= train_test_split(df.drop('Class label',axis =1),df['Class label'],test_size=0.3)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train_scaled = pd.DataFrame(X_train_scaled,columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled,columns=X_test.columns)

fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize=(12,5))
ax1.scatter(X_train['Alcohol'],X_train['Malic acid'],c = Y_train)
ax1.set_title("Before Scaling")
ax2.scatter(X_train_scaled['Alcohol'],X_train_scaled['Malic acid'],c = Y_train)
ax2.set_title("After Scaling")
plt.show()