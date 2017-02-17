#Import iris dataset
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 
import numpy as np 

iris = datasets.load_iris()

x = iris.data
y = iris.target

tree_classifier = tree.DecisionTreeClassifier()
neighbors_classifier = KNeighborsClassifier()

accuracy_tree = []
accuracy_neighbors = []

for i in range(1,2000):
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)
	tree_classifier.fit(x_train, y_train)
	tree_predictions = tree_classifier.predict(x_test)
	neighbors_classifier.fit(x_train, y_train)
	neighbors_predictions = neighbors_classifier.predict(x_test)
	accuracy_tree.append(accuracy_score(y_test, tree_predictions))
	accuracy_neighbors.append(accuracy_score(y_test, neighbors_predictions))


print("min accuracy tree: ", min(accuracy_tree))
print("max accuracy tree: ", max(accuracy_tree))
print("avg accuracy tree: ", np.mean(accuracy_tree))

print("min accuracy neighbors: ", min(accuracy_neighbors))
print("max accuracy neighbors: ", max(accuracy_neighbors))
print("avg accuracy neighbors: ", np.mean(accuracy_neighbors))

plt.hist([accuracy_tree, accuracy_neighbors], stacked = True, color = ['r','b'])
plt.show()

