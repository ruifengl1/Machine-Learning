import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))

def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row. (Convert 0 to -1)
    """
    with open(filename, 'r') as fp:
        array2d = np.array([[digit for digit in line.split(',')] for line in fp], dtype=np.float32)
    Y = array2d[:,-1]
    Y = np.where(Y == 0, -1., 1.)
    X = array2d[:, :-1]
    
    return X, Y


def adaboost(X, y, num_iter, max_depth=1):
    """Given an numpy matrix X, a array y and num_iter return trees and weights 
   
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is {-1, 1}
    """
    trees = []
    trees_weights = [] 
    N, _ = X.shape
    w = np.ones(N) / N

    for i in range(num_iter):
        # fit tree
        h = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        h.fit(X, y, sample_weight=w)
        
        # compute error
        y_preds = h.predict(X)
        err = np.sum(w[~(y_preds == y)])/np.sum(w)
        alpha = np.log((1-err)/err)
        w[~(y_preds == y)] *= np.exp(alpha)
        trees.append(h)
        trees_weights.append(alpha)
    
    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y
    """
    # X input, y output
    N, _ =  X.shape
    y = np.zeros(N)
    for i, h in enumerate(trees):
        y += h.predict(X) * trees_weights[i]
    y = np.sign(y)
    
    return y
