import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')

#Getting the information of our data
print(df.info())

#Filling the data 
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(["Ticket","Name","PassengerId","Cabin"], axis=1,inplace=True)

#Converting the categoricalData to Digits
df['Sex'] = df['Sex'].map({'male':0, "female":1})
df['Embarked'] = df['Embarked'].map({'S':0, "C":1,"Q":2})

#Splitting and training
X = df.drop("Survived",axis=1)
Y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#Plotting

sns.countplot(data=df, x='Survived', hue='Sex')
plt.title("Survival Count by Gender")
plt.show()
