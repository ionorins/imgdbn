import numpy as np
import cv2
import os

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from dbn.models import UnsupervisedDBN
from sklearn.neighbors import KNeighborsClassifier

path = '../flower_photos/'

i = 0
X = []
Y = []

for folder in os.listdir(path):
	for image in os.listdir(path + folder):
		img = cv2.imread(path + folder + '/' + image) # , cv2.IMREAD_GRAYSCALE
		img = cv2.resize(img, (32, 32)) 
		X.append(img)
		Y.append(i)
	i += 1
     

X = np.asarray(X)
Y = np.asarray(Y)
np.random.seed(7)

X = X.reshape(len(X), X.size // len(X))
X = X / 255

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

from sklearn.model_selection import GridSearchCV
#create new a knn model
knn2 = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {
    # 'n_neighbors': np.arange(1, 25),
              'weights' : ['distance'],
            #   'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p' : np.arange(1,6)
              }
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X, Y)

print(knn_gscv.best_params_)
print(knn_gscv.best_score_)