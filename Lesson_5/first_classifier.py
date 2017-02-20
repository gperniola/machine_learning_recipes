import random

from scipy.spatial import distance

def euc(a, b):
	return distance.euclidean(a, b)

#Creating EuclideanKNN
class euclideanKNN():
	def fit(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train

	def predict(self, x_test):
		predictions = []
		for row in x_test:
			label = self.closest(row)
			predictions.append(label)
		return predictions

	def closest(self, row):
		best_dist = euc(row, self.x_train[0])
		best_index = 0
		for i in range(1, len(self.x_train)):
			dist = euc(row, self.x_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i
		return self.y_train[best_index]




#Creating RandomKNN
class randomKNN():
	def fit(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train

	def predict(self, x_test):
		predictions = []
		for row in x_test:
			label = random.choice(self.y_train)
			predictions.append(label)
		return predictions


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


#random classifier
random_classifier = randomKNN()
random_classifier.fit(x_train, y_train)
random_predictions = random_classifier.predict(x_test)

print ("random classifier accuracy:", accuracy_score(y_test, random_predictions))


#euclidean classifier
euclidean_classifier = euclideanKNN()
euclidean_classifier.fit(x_train, y_train)
euclidean_predictions = euclidean_classifier.predict(x_test)

print ("euclidean classifier accuracy:", accuracy_score(y_test, euclidean_predictions))