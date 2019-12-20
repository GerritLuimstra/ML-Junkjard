from util.criterions import RSS, RMSE
from DecisionTreeRegression.regression_tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import math

"""
    Please note that this is not a *gradient* boosted regression tree,
    however it does utilize boosting!
"""


class BoostedRegressionTree:

    def __init__(self, n_estimators, min_leaf_size=2, learning_rate=0.01, max_splits=None, learning_rate_decay=True, debug=False):
        self.debug = debug
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_splits = max_splits
        self.min_leaf_size = min_leaf_size
        self.trees = []
        self.residual_mean = 0
        self.learning_rate_decay = learning_rate_decay

    """
        Simple learning rate scheduler
    """
    def learning_rate_scheduler(self, current_learning_rate, iteration):
        drop = 0.99
        iteration_drop = max(1, self.n_estimators // 10)
        return current_learning_rate * drop ** math.floor(iteration / iteration_drop)

    """
        Fits a boosted regression tree based on the features, criterion and responses
    """
    def fit(self, X, y, criterion):

        # Setup the correct learning rate
        learning_rate = self.learning_rate

        # Compute the base residuals
        residuals = y

        for i in range(self.n_estimators):

            if self.debug:
                print("FITTING TREE {}/{} ".format(i+1, self.n_estimators))

            # Fit a tree to the residuals
            tree = DecisionTreeRegressor(min_leaf_size=self.min_leaf_size, max_splits=self.max_splits)
            tree.fit(X, residuals, criterion)

            # Obtain the predictions of the tree trained on the residuals
            predictions = tree.predict(X)

            # Update the residuals
            residuals -= self.learning_rate * np.array(predictions)

            # Add the 'shrunken' tree to the current trees
            self.trees.append(tree)

            if self.learning_rate_decay:
                # Update the learning rate
                learning_rate = self.learning_rate_scheduler(learning_rate, i)

            print(RMSE(self.predict(X), y))

    """
        Given an input X, compute the response Y
    """
    def predict(self, X):

        learning_rate = self.learning_rate

        output = np.array([0] * len(X))
        for i, tree in enumerate(self.trees):
            output = np.add(output, self.learning_rate * np.array(tree.predict(X)))

            if self.learning_rate_decay:
                # Update the learning rate
                learning_rate = self.learning_rate_scheduler(learning_rate, i)

        return output


if __name__ == "__main__":

    dataset = pd.read_csv("../DecisionTreeRegression/Hitters.csv", index_col="Unnamed: 0")
    dataset = dataset.dropna()

    feature_names = ["Years", "Hits"] # set(dataset.columns) - set(["Salary"])
    features = dataset[feature_names]
    response = dataset["Salary"]
    response = np.log10(response)

    # Define the boosted regressor
    regressor = BoostedRegressionTree(min_leaf_size=5, learning_rate=0.03, max_splits=5, n_estimators=50, debug=True)
    regressor.fit(features, response, RSS)

    # Obtain the predictions
    predictions = regressor.predict(features)

    # Calculate the RMSE
    predictions = np.array(predictions)
    actuals = np.array(response)

    # Print out the RMSE
    print(RMSE(actuals, predictions))