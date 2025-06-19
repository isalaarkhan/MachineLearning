from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

df = load_iris()
x = pd.DataFrame(df.data, columns=df.feature_names)
y = df.target

df = x.copy()
df['target'] = y
sns.pairplot(df,hue= 'target')
#plt.show()

X_train, X_test,Y_train, Y_test = train_test_split(x,y,test_size=0.2)

sts = StandardScaler()
X_train_sts = sts.fit_transform(X_train)
X_test_sts = sts.fit_transform(X_test)

model = SVC(kernel='linear')
model.fit(X_train_sts,Y_train)

y_pred = model.predict(X_test_sts)


print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
print("Classification Report:\n", classification_report(Y_test, y_pred))