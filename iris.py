import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

print("- print names of features and target")
print(iris.feature_names)
print(iris.target_names)

print("\n- print first element data and target")
print(iris.data[0])
print(iris.target[0])

#prints all the dataset
#for i in range(len(iris.target)):
#	print ("Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))

#getting test data indexes from dataset..
test_idx = [1,51,101]

#training data (splitting target from data)
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#training classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

#making prediction
prediction = clf.predict(test_data)

print("\n- Printing test data and prediction...")
for i in range(len(test_target)):
	print("Example %d: feats: %s, label: %s, prediction: %s" %(i, test_data[i],test_target[i],prediction[i]))


#from sklearn.externals.six import StringIO
#import pydot
#dot_data = tree.export_graphviz(clf, out_file=None, 
#                         feature_names=iris.feature_names,  
#                         class_names=iris.target_names,  
#                         filled=True, rounded=True,  
#                         special_characters=True) 
#graph = pydotplus.graph_from_dot_data(dot_data) 
#graph.write_pdf("iris.pdf") 








