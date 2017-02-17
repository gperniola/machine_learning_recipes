#Import iris dataset
from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

#Partitioning data in training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

#Using different classifiers
#Tree classifier
from sklearn import tree 
tree_classifier = tree.DecisionTreeClassifier()
tree_classifier.fit(x_train, y_train)
tree_predictions = tree_classifier.predict(x_test)

#Calculating accuracy of tree classifier
from sklearn.metrics import accuracy_score
print ("Tree classifier accuracy:", accuracy_score(y_test, tree_predictions))


#K Neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
neighbors_classifier = KNeighborsClassifier()
neighbors_classifier.fit(x_train, y_train)
neighbors_predictions = neighbors_classifier.predict(x_test)

#Calculatin accuracy of KNeighbors classifier
print ("KNeighbors classifier accuracy:", accuracy_score(y_test, neighbors_predictions))