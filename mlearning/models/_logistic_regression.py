import numpy as np 
from mlearning.activations import sigmoid

class LogisticRegression:

    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    
    def fit(self, X, y, verbose=False):
        training_size = X.shape[0]
        X = np.concatenate(
            (
                np.ones((training_size, 1)),
                X
            ), axis=1
        )
        y = np.transpose([y])

        params = np.zeros((X.shape[1], 1))
        
        self.loss_ = []
        iteration = 1

        while True:
            hypothesis = sigmoid(X.dot(params))
            self.loss_.append(self.calculate_loss(hypothesis, y))

            if verbose:
                print(f"Itration {iteration}. Loss function = {self.loss_[-1]:.9f}")
                iteration += 1

            d_J = (X.T @ (hypothesis - y)) / training_size
    
            if len(self.loss_) > 1 and abs(self.loss_[-1] - self.loss_[-2]) < 1e-9:
                break
            
            params = params - self.learning_rate * d_J
        
        self.params_ = params


    def calculate_loss(self, hypothesis, y):
        return (- 1 / hypothesis.shape[0]) * np.sum(y * np.log(hypothesis) + (1 - y) * np.log(1 - hypothesis))


    def predict(self, X):
        probs = self.predict_probabilities(X)
        return [ 1 if prob >= 0.5 else 0 for prob in probs]

    def predict_probabilities(self, X):
        X = np.concatenate(
            (
                np.ones((X.shape[0], 1)),
                X
            ), axis=1
        )
        return sigmoid(X @ self.params_).reshape((X.shape[0],))