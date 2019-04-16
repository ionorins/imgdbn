import numpy as np
import cv2
import os

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from dbn.models import UnsupervisedDBN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston, load_iris, load_diabetes, load_digits, load_linnerud, load_wine, load_breast_cancer

scaler = MinMaxScaler()
datasets = load_iris, load_digits, load_wine, load_breast_cancer
     
for dataset in datasets:
    data = dataset()
    X = np.asarray(data.data, 'float32')
    Y = np.asarray(data.target)

    X = X.reshape(len(X), X.size // len(X))
    X = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    # Models we will use
    knn = KNeighborsClassifier(weights = 'distance', n_jobs=-1) 
    dbn = UnsupervisedDBN(hidden_layers_structure=[256, 512],
                      batch_size=16,
                      n_epochs_rbm=50,
                      activation_function='sigmoid',
                      verbose=False)


    classifier = Pipeline(steps=[('dbn', dbn),
                                 ('knn', knn)])

    classifier.fit(X_train, Y_train)

    knn_classifier = KNeighborsClassifier(n_jobs=-1)
    knn_classifier.fit(X_train, Y_train)

    ###############################################################################
    # Evaluation

    print()
    print("KNN using RBM features:\n%s\n" % (
        metrics.classification_report(
            Y_test,
            classifier.predict(X_test))))

    print("KNN regression using raw pixel features:\n%s\n" % (
        metrics.classification_report(
            Y_test,
            knn_classifier.predict(X_test))))

    ###############################################################################