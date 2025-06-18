from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Try different K values
for k in range(1, 105):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    print(f"K = {k}, Accuracy = {accuracy:.2f}")
