import pandas as pd
from  sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df = pd.read_csv('titanic.csv')
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1,inplace=True)

num_imputer = SimpleImputer(strategy='median')  
cat_imputer = SimpleImputer(strategy='most_frequent')  

df['Age'] = num_imputer.fit_transform(df[['Age']])
df['Embarked'] = cat_imputer.fit_transform(df[['Embarked']]).ravel()
#print(df.isnull().sum())

df = pd.get_dummies(df,columns=['Sex','Embarked'],drop_first=True)

x = df.drop('Survived',axis=1)
y = df['Survived']


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)

y_pred = dt_model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))