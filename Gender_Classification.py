from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np


clf=tree.DecisionTreeClassifier()
knn=KNeighborsClassifier(n_neighbors=3)
nb =  GaussianNB()
sv= SVC(kernel="linear", C=0.025)

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

X_test = [[184,84,44],[198,92,48],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]


# Fitting of Data
clf = clf.fit(X, Y)
knn= knn.fit(X,Y)
nb = nb.fit(X,Y)
sv= sv.fit(X,Y)

pred_tree =   clf.predict(X_test)
pred_knn   =   knn.predict(X_test)
pred_nb    =   nb.predict(X_test)
pred_sv    =   sv.predict(X_test)
# CHALLENGE compare their reusults and print the best one!
	
#print(clf.score(X_test,pred_tree),knn.score(X_test,pred_knn) , nb.score(X_test,pred_nb) , sv.score(X_test,pred_sv)  )
index = np.argmax([clf.score(X_test,pred_tree),knn.score(X_test,pred_knn) , nb.score(X_test,pred_nb) , sv.score(X_test,pred_sv)])
classifiers = {0: 'Decision Tree', 1: 'K Nearest Neighbour', 2: 'Gaussian Naive Bayes', 3: 'Linear Support Vector Machine'}
print('Best gender classifier is {}'.format(classifiers[index]))
