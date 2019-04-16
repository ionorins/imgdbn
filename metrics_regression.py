import numpy as np
import cv2
import os

from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from dbn.models import UnsupervisedDBN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston, load_iris, load_diabetes, load_digits, load_linnerud, load_wine, load_breast_cancer

scaler = MinMaxScaler()
datasets = load_boston, load_diabetes, load_linnerud
     
for dataset in datasets:
    data = dataset()
    X = np.asarray(data.data, 'float32')
    Y = np.asarray(data.target)

    X = X.reshape(len(X), X.size // len(X))
    X = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    # Models we will use
    knn = KNeighborsRegressor(weights = 'distance', n_jobs=-1) 
    dbn = UnsupervisedDBN(hidden_layers_structure=[256, 512],
                      batch_size=10,
                      learning_rate_rbm=0.06,
                      n_epochs_rbm=20,
                      activation_function='sigmoid')


    classifier = Pipeline(steps=[('dbn', dbn),
                                 ('knn', knn)])

    # classifier.fit(X_train, Y_train)

    knn_classifier = KNeighborsRegressor(n_jobs=-1)
    # knn_classifier.fit(X_train, Y_train)

    ###############################################################################
    # Evaluation
    # Y_pred = classifier.predict(X_test)
    # print(metrics.average_precision_score(Y_test,Y_pred))
    # print(metrics.recall_score(Y_test,Y_pred))
    # print(metrics.precision_score(Y_test,Y_pred))
    print()
    print("KNN using RBM features:")
    kfold = KFold(n_splits=5)
    scoring = 'neg_mean_absolute_error'
    results = cross_val_score(classifier, X, Y, cv=kfold)
    print(("MAE: %.3f (%.3f)") % (results.mean(), results.std()))

    print("KNN using RBM features:")
    results = cross_val_score(knn_classifier, X, Y, cv=kfold)
    print(("MAE: %.3f (%.3f)") % (results.mean(), results.std()))
    # print(metrics.f1_score(Y_test,Y_pred))
    # print()
    # print("KNN using RBM features:\n%s\n" % (
    #     metrics.precision_recall_fscore_support(
    #         Y_test,
    #         classifier.predict(X_test))))

    # print("KNN regression using raw pixel features:\n%s\n" % (
    #     metrics.precision_recall_fscore_support(
    #         Y_test,
    #         knn_classifier.predict(X_test))))

    ###############################################################################