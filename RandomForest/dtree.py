import numpy as np

class LeafNode:
    def __init__(self, y, prediction):
        """
        Create leaf node from y values and prediction; 
        Prediction is mean(y) or mode(y)
        """
        self.n = len(y)
        self.prediction = prediction

    def leaf(self, x_test):
        return self

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def leaf(self, x_test):
        """
        Given a single test record, x_test, return the leaf node reached by running it down the tree starting at this node.
        """
        if x_test[self.col] <= self.split:
            return self.lchild.leaf(x_test)
        return self.rchild.leaf(x_test)

class DecisionTree621:
    def __init__(self, min_samples_leaf=1, max_features=0.3, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
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
        col, split = self.bestsplit(X, y)
        if col == -1: 
            return self.create_leaf(y)
        lchild = self.fit_(X[X[:, col]<=split], y[X[:, col] <=split])
        rchild = self.fit_(X[X[:, col]>split], y[X[:, col]>split])
        return DecisionNode(col, split, lchild, rchild)

    def bestsplit(self, X, y):
        best = (-1, -1, self.loss(y))
        k = 11
        vars = np.random.choice(X.shape[1], round(self.max_features * X.shape[1]), replace=False)
        for col in vars:
            rand_pick = np.random.choice(X.shape[0], k if len(X) >=k else X.shape[0], replace=False)
            candidates = X[:, col][rand_pick]
            for split in candidates:
                yl = y[X[:, col] <= split]
                yr = y[X[:, col] > split]
                if yl.shape[0] < self.min_samples_leaf or yr.shape[0] < self.min_samples_leaf:
                    continue
                l = (yl.shape[0] * self.loss(yl) + yr.shape[0] * self.loss(yr))/y.shape[0]
                if l == 0:
                    return col, split
                if l < best[2]:
                    best = (col, split, l)
                
        return best[0], best[1]


    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        """
        y_pred = [self.root.leaf(i).prediction for i in X_test]
        return np.array(y_pred)

class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1, max_features=0.3):
        super().__init__(min_samples_leaf, max_features, loss=np.std)

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))
        

class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1, max_features=0.3):
        super().__init__(min_samples_leaf,max_features, loss=gini)

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