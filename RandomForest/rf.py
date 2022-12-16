import numpy as np
from sklearn.utils import resample
from dtree import *

class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different, bootstrapped versions of the training data. Keep track of the indexes of the OOB records for each tree.
        """
        # compute/store the number of unique y values
        self.nunique = len(np.unique(y))

        # fit decision trees
        self.trees = list()
        for i in range(self.n_estimators):
            indices = resample (range(X.shape[0]), n_samples= X.shape[0])
            X_, y_ = X[indices], y[indices]
            tree = self.initialize_tree(X_, y_)
            self.trees.append(tree)
            
            # find the oob index
            oob_index = set(range(X.shape[0])) - set(indices)
            tree.oob_idxs = np.array(list(oob_index))

        # calculate oob_score
        if self.oob_score:
            self.oob_score_ = self.compute_oob_score(X, y)


class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def compute_oob_score(self, X, y):
        oob_counts = np.zeros(X.shape[0], dtype=np.int64)
        oob_preds = np.zeros(X.shape[0], dtype=np.float64)

        for t in self.trees:
            leafsizes = np.array([t.root.leaf(i).n for i in X[t.oob_idxs]])
            oob_preds[t.oob_idxs] += leafsizes * t.predict(X[t.oob_idxs])
            oob_counts[t.oob_idxs] +=leafsizes
        
        oob_avg_preds = oob_preds[oob_counts > 0] / oob_counts[oob_counts > 0]

        corr_matrix = np.corrcoef(y[oob_counts > 0], oob_avg_preds)
        corr = corr_matrix[0,1]
        R_sq = corr**2
        return R_sq

    def initialize_tree(self, X ,y):
        tree = RegressionTree621(self.min_samples_leaf, self.max_features)
        tree.fit(X, y)
        return tree



    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average prediction from all trees in this forest.
        """
        n_obs = np.zeros(X_test.shape[0], dtype=np.int64)
        y_sum = np.zeros(X_test.shape[0], dtype=np.float64)
        for t in self.trees:
            n = []
            y = []
            for i in range(X_test.shape[0]):
                leaf = t.root.leaf(X_test[i])
                n.append(leaf.n)
                y.append(leaf.prediction * leaf.n)

            n_obs += np.array(n)
            y_sum += np.array(y)

        return y_sum / n_obs
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records, collect the prediction for each record and then compute R^2 on that and y_test.
        """
        y_pred = self.predict(X_test)
        corr_matrix = np.corrcoef(y_test, y_pred)
        corr = corr_matrix[0,1]
        R_sq = corr**2
        return R_sq

class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def compute_oob_score(self, X, y):
        oob_counts = np.zeros(X.shape[0], dtype=np.int64)
        oob_preds = np.zeros((X.shape[0], self.nunique), dtype=np.int64)
        oob_votes = np.zeros(X.shape[0], dtype=np.int64)

        for t in self.trees:
            leafsizes = np.array([t.root.leaf(i).n for i in X[t.oob_idxs]])
            tpred = t.predict(X[t.oob_idxs])
            oob_preds[t.oob_idxs, tpred] += leafsizes
            oob_counts[t.oob_idxs] += 1
        for i in np.where(oob_counts > 0):
            oob_votes[i] = np.argmax(oob_preds[i, :], axis=1)

        accuracy_score = np.sum(y[oob_counts >0] == oob_votes[oob_counts >0]) / y[oob_counts >0].shape[0]
        return accuracy_score


    def initialize_tree(self, X, y):
        tree = ClassifierTree621(self.min_samples_leaf, self.max_features)
        tree.fit(X, y)
        return tree

    def predict(self, X_test) -> np.ndarray:
        total_counts = []
        for i in X_test:
            counts = np.zeros(self.nunique, dtype=np.int64)
            for t in self.trees:
                leaf = t.root.leaf(i)
                counts[leaf.prediction] +=1
            total_counts.append(counts)
        
        return np.argmax(total_counts, axis=1)
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records, collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        y_pred = self.predict(X_test)
        accuracy_score = (y_test == y_pred).sum() / y_test.shape[0]
        return accuracy_score

# pytest -v -n 8 test_rf.py