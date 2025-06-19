from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train,X_test,Y_train,Y_test = train_test_split(X,y, test_size=0.2)

sts= StandardScaler()
X_test_sts =sts.fit_transform(X_test)
X_train_sts =sts.fit_transform(X_train)

model = SVC(kernel='linear')
model.fit(X_train_sts,Y_train)

y_pred = model.predict(X_test_sts)

print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
print("Classification Report:\n", classification_report(Y_test, y_pred))