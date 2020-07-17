import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

class LinearRegression:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate


    def fit(self, X, y, verbose=False):
        training_size = X.shape[0]
        params = np.zeros((X.shape[1] + 1, 1))
        X = np.concatenate((np.ones((training_size, 1)), X), axis=1)
        y = np.transpose([y])
        self.loss_ = []
        iteration = 1

        while True:
            hypothesis = np.dot(X, params)
            d_J = X.T.dot((hypothesis - y)) / training_size

            self.loss_.append(np.sum((hypothesis - y) ** 2) / (2 * training_size) )
            
            if verbose:
                print(f'Iteration {iteration}. Loss Function: {self.loss_[-1]:.5f}.')
                iteration += 1

            if len(self.loss_) > 1 and abs(self.loss_[-1] - self.loss_[-2]) < 1e-15:
                break
            
            params = params - self.learning_rate * d_J

        self.params_ = params


    def predict(self, X):
        return np.dot( 
                np.concatenate((np.ones((X.shape[0], 1)), X), axis=1), 
                self.params_
            ).reshape((X.shape[0],))


