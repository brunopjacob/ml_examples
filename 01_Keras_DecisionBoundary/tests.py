# Import packages

import numpy as np
np.random.seed(0)
from sklearn import datasets
from sklearn import preprocessing
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization


X, Y = datasets.make_moons(n_samples=10, noise=0.0, random_state=0)

plt.figure(1)
colors = ['blue' if label == 1 else 'red' for label in Y]
plt.scatter(X[:,0], X[:,1], color=colors)

print(X)

# Use preprocessing to re-scale data
#X = preprocessing.scale(X)
mm_scaler = preprocessing.StandardScaler()
X = mm_scaler.fit_transform(X)
mm_scaler.transform(X)


#Y = preprocessing.scale(Y)

plt.figure(2)
plt.scatter(X[:,0], X[:,1], color=colors)

plt.show()

print(X)
