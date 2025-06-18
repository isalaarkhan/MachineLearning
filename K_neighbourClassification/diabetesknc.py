import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


df = pd.read_csv('diabetes.csv')
#print(df.head())
#print(df.isnull().sum())

X = df.iloc[:,0:8]
Y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

sts = StandardScaler()
X_test_sts = sts.fit_transform(X_test)
X_train_sts = sts.fit_transform(X_train)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_sts,y_train)

y_pred = knn.predict(X_test_sts)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
