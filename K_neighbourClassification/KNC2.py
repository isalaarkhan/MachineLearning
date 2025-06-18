import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

x = [[180, 80], [160, 60], [170, 70], [155, 50], [185, 90]]
y = ['male', 'female', 'male', 'female', 'male']

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x,y)

print(clf.predict([[165,55]]))