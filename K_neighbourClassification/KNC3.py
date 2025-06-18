from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=2, n_redundant=0)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
plt.title("KNN Classification Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
