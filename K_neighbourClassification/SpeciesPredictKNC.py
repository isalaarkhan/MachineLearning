from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


iris = load_iris()
x = iris.data
y = iris.target 


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(x_train, y_train)

print("Model Accuracy:", clf.score(x_test, y_test))
print("Predicted:", clf.predict(x_test[:20]))
print("Actual:   ", y_test[:20])
