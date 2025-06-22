import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
df = pd.read_csv('D:\ML\CODES\Basics\loan_data.csv')
df = df.drop('Loan_ID',axis=1)

cat_colums = df.select_dtypes(include='object').columns.tolist()
cat_colums.remove('Loan_Status')
num_colums = df.select_dtypes(include=['int64','float64']).columns.tolist()

cat_imput = SimpleImputer(strategy='most_frequent')
num_imput = SimpleImputer(strategy='median')
df[cat_colums] = cat_imput.fit_transform(df[cat_colums])
df[num_colums] = num_imput.fit_transform(df[num_colums])


df['Loan_Status'] = df['Loan_Status'].map({'Y':1,'N':0})
df = pd.get_dummies(df,columns=cat_colums,drop_first=True)


X = df.drop('Loan_Status',axis=1)
Y = df['Loan_Status']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

model = DecisionTreeClassifier()

model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))