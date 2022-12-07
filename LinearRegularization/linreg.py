import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def normalize(X):
    """
    Creating standard variables here (u-x)/sigma
    :param X:
    :return:
    """
    if isinstance(X, pd.DataFrame):
        for c in X.columns:
            if is_numeric_dtype(X[c]):
                u = np.mean(X[c])
                s = np.std(X[c])
                X[c] = (X[c] - u) / s
        return
    for j in range(X.shape[1]):
        u = np.mean(X[:,j])
        s = np.std(X[:,j])
        X[:,j] = (X[:,j] - u) / s

def loss_gradient(X, y, B, lmbda):
    return np.dot(-X.T, y - np.dot(X, B))

def loss_ridge(X, y, B, lmbda):
    return np.dot(y - np.dot(X , B), y - np.dot(X, B)) + np.dot((lmbda * B), B)

def loss_gradient_ridge(X, y, B, lmbda):
    return loss_gradient(X, y, B, lmbda) + lmbda * B

def sigmoid(z):
    return 1/(1+np.exp(-z))

def log_likelihood(X, y, B,lmbda):
    return -np.sum(np.dot(y, np.dot(X, B)) - np.log(1+ np.exp(np.dot(X, B))))

def log_likelihood_gradient(X, y, B, lmbda):
    return np.dot(-X.T, y - sigmoid(np.dot(X, B)))

def L1_log_likelihood(X, y, B, lmbda):
    z = np.dot(X, B)
    return -1 * np.sum(np.dot(y, B) - np.log(1 + np.exp(z))) + lmbda * np.sum(np.abs(B))

def L1_log_likelihood_gradient(X, y, B, lmbda):
    n, p = X.shape
    z = np.dot(X, B)
    err = y - sigmoid(z)
    r = lmbda * np.sign(B)
    r[0] = 0
    grad = (np.dot(np.transpose(X), err) - r) / n
    return B - grad

def minimize(X, y, loss_gradient,
              eta=0.00001, lmbda=0.0,
              max_iter=1000, addB0=True,
              precision=1e-9):
    "Here are various bits and pieces you might want"
    if X.ndim != 2:
        raise ValueError("X must be n x p for p features")
    n, p = X.shape
    if y.shape != (n, 1):
        raise ValueError(f"y must be n={n} x 1 not {y.shape}")

    if addB0:
        row_num = X.shape[0]
        ones = np.ones((row_num,1))
        X = np.concatenate((ones, X), axis=1)
        p = p + 1

    h = 0
    B = np.random.random_sample(size=(p, 1)) * 2 - 1  # make between [-1,1)
    eps = 1e-5 # prevent division by 0
    for i in range(max_iter):
        gradient = loss_gradient(X, y , B, lmbda)
        h += np.multiply(gradient, gradient)
        B = B - eta * gradient/(np.sqrt(h) + eps)
        if np.linalg.norm(gradient) < precision:
            return B
    return B


class LinearRegression621:
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                          loss_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter)


class LogisticRegression621:
    def __init__(self, eta=0.00001, lmbda=0.0, max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict_proba(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        z = np.dot(X, self.B)
        return sigmoid(z)

    def predict(self, X):
        y_pred_prob = self.predict_proba(X)
        y_pred = y_pred_prob.copy()
        y_pred[y_pred_prob >= 0.5] = 1
        y_pred[y_pred_prob < 0.5] = 0
        return y_pred

    def fit(self, X, y):
        self.B = minimize(X, y,
                          log_likelihood_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter)


class RidgeRegression621:
    def __init__(self, eta=0.00001, lmbda=0.0, max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        ones = np.ones(shape=(n, 1))
        X = np.hstack([ones, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B0 = np.mean(y)
        self.B = minimize(X, y,
                        loss_gradient_ridge,
                        self.eta,
                        self.lmbda,
                        self.max_iter, addB0=False)
        self.B = np.vstack([self.B0, self.B])


class LassoLogistic621:
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict_proba(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        z = np.dot(X, self.B)
        return sigmoid(z)

    def predict(self, X):
        prob = self.predict_proba(X)
        pred = []
        for i in prob.flatten():
            if i > 0.5:
                pred.append(1)
            else:
                pred.append(0)
        pred = np.array(pred).reshape(-1, 1)
        return pred

    def fit(self, X, y):
        self.B = minimize(X, y,
                          L1_log_likelihood_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter,
                          addB0=False)
        z = np.dot(X, self.B)
        err = y - sigmoid(z)
        beta_b0 = np.array(np.mean(err))
        self.B = np.vstack([beta_b0, self.B]).reshape(-1, 1)

# python -m pytest -v --count=20 test_regr.py
# python -m pytest -v --count=20 test_class.py