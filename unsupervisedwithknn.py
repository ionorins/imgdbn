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
     

# Load Data
# digits = datasets.load_digits()
# X = np.asarray(digits.data, 'float32')
# Y = digits.target

X = np.asarray(X)
Y = np.asarray(Y)

X = X.reshape(len(X), X.size // len(X))
X = X / 255

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# Models we will use
knn = KNeighborsClassifier(weights = 'distance') 
dbn = UnsupervisedDBN(hidden_layers_structure=[24, 16, 8],
                      batch_size=16,
                      learning_rate_rbm=0.06,
                      n_epochs_rbm=500,
                      activation_function='sigmoid')


classifier = Pipeline(steps=[('dbn', dbn),
                             ('knn', knn)])

classifier.fit(X_train, Y_train)

# Training Logistic regression
knn_classifier = KNeighborsClassifier(weights = 'distance')
knn_classifier.fit(X_train, Y_train)

###############################################################################
# Evaluation

# print()
print("KNN using RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        classifier.predict(X_test))))

print("KNN regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        knn_classifier.predict(X_test))))

###############################################################################
