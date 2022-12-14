import numpy as np

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if x_test[self.col] <= self.split:
            return self.lchild.predict(x_test)
        return self.rchild.predict(x_test)

class LeafNode:
    def __init__(self, y, prediction):
        """
        Create leaf node from y values and prediction; prediction is mean(y) or mode(y)
        """
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        return self.prediction

class DecisionTree621:
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.std or gini

    def fit(self, X, y):
        """
        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)
        
    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classifier or regressor.
        """
        if X.shape[0]<= self.min_samples_leaf: # if one x value, make leaf
            return self.create_leaf(y)
        col, split = self.bestsplit(X, y, self.loss)
        if col == -1: 
            return self.create_leaf(y)
        lchild = self.fit_(X[X[:, col]<=split], y[X[:, col] <=split])
        rchild = self.fit_(X[X[:, col]>split], y[X[:, col]>split])
        return DecisionNode(col, split, lchild, rchild)

    def bestsplit(self, X, y, loss):
        best = (-1, -1, loss(y))
        k = 11
        for col in range(X.shape[1]):
            rand_pick = np.random.choice(X.shape[0], k if len(X) >=k else X.shape[0], replace=False)
            candidates = X[:, col][rand_pick]
            for split in candidates:
                yl = y[X[:, col] <= split]
                yr = y[X[:, col] > split]
                if yl.shape[0] < self.min_samples_leaf or yr.shape[0] < self.min_samples_leaf:
                    continue
                l = (yl.shape[0] * loss(yl) + yr.shape[0] * loss(yr))/y.shape[0]
                if l == 0:
                    return col, split
                if l < best[2]:
                    best = (col, split, l)
                
        return best[0], best[1]
    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621
        """
        y_pred = [self.root.predict(X_test[i]) for i in range(X_test.shape[0])]
        return y_pred

class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.std)
    def score(self, X_test, y_test):
        """
        Return the R^2 of y_test vs predictions for each record in X_test
        """
        y_pred = self.predict(X_test)
        corr_matrix = np.corrcoef(y_test, y_pred)
        corr = corr_matrix[0,1]
        R_sq = corr**2
        return R_sq

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))
        

class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini)
    def score(self, X_test, y_test):
        """
        Return the accuracy_score() of y_test vs predictions for each record in X_test
        """
        y_pred = self.predict(X_test)
        accuracy_score = (y_test == y_pred).sum() / y_test.shape[0]
        return accuracy_score

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor.
        """
        vals,counts = np.unique(y, return_counts=True)
        index = np.argmax(counts)
        return LeafNode(y, vals[index])

def gini(y):
    """
    Return the gini impurity score for values in y
    """
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum( p**2 )

# python -m pytest -v test_dtree.py