import numpy as np
import cv2
import os
from scipy.misc import imsave



from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from dbn.models import UnsupervisedDBN
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits

scaler = MinMaxScaler()

X = np.asarray(load_digits().data, 'float32')
Y = np.asarray(load_digits().target)

X_copy = X[:]
X = X.reshape(len(X), X.size // len(X))
X = scaler.fit_transform(X)

# Models we will use
knn = KMeans(n_clusters=10, n_jobs=-1)
dbn = UnsupervisedDBN(hidden_layers_structure=[256, 512],
                    batch_size=16,
                    n_epochs_rbm=20,
                    activation_function='sigmoid',
                    verbose=False)


classifier = Pipeline(steps=[('dbn', dbn),
                                ('knn', knn)])

classifier.fit(X)

pred = classifier.predict(X)

X_copy = X_copy.reshape(len(X_copy), 8, 8)
for i in range(len(pred)):
    imsave(str(pred[i]) + '_' + str(i) + '.png', X_copy[i])

