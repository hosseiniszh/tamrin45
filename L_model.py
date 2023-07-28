import numpy as np 
from numpy.linalg import inv
class LLS:
    def __init__(self):
        self.w = None
    def fit(self, X_train, Y_train):
        self.w = inv(X_train.T @ X_train) @ X_train.T @ Y_train
    def predict(self, X_test):
        return np.matmul(X_test, self.w)
