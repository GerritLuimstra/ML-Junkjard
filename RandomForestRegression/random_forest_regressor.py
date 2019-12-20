from sklearn.utils import resample
from DecisionTreeRegression.regression_tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import math
from random import choices
import random


class RandomForestRegressor:

    def __init__(self, n_trees, min_leaf_size, n_features, decorrelate_strategy="sqrt"):
        self.n_trees = n_trees
        self.min_leaf_size = min_leaf_size
        self.trees = []

        # Determine the wanted feature count
        if decorrelate_strategy == "sqrt":
            self.n_features = math.ceil(n_features**(1/2))
        elif decorrelate_strategy == "div3":
            self.n_features = math.ceil(n_features / 3)
        else:
            self.n_features = n_features

    """
        Fits a random forest of decision trees based on the features and responses
    """
    def fit(self, features, response, criterion_func):

        for _ in range(self.n_trees):

            # Obtain a new batch of bootstrap samples
            bagged_indices = choices(features.index, k=len(features))
            bagged_samples = features.loc[bagged_indices]
            bagged_response = response.loc[bagged_indices]

            # Reindex the bagged_samples and bagged_response to prevent errors further on
            bagged_samples = bagged_samples.reset_index()
            bagged_response = bagged_response.reset_index()
            bagged_samples = bagged_samples.drop(["index"], axis=1)
            bagged_response = bagged_response.drop(["index"], axis=1)
            bagged_response = bagged_response[bagged_response.columns[0]]

            # Obtain the features to be used
            features_to_use = random.sample(list(features.columns), self.n_features)

            # Only make it use the selected features
            bagged_samples = bagged_samples[features_to_use]

            # Define the decision tree regressor
            regressor = DecisionTreeRegressor(min_leaf_size=self.min_leaf_size)

            # Train on the bagged samples
            regressor.fit(bagged_samples, bagged_response, criterion_func)

            # Add the tree to the list
            self.trees.append(regressor)

    """
        Given an input X, compute the response Y
    """
    def predict(self, X):

        # Obtain the predictions for each tree
        predictions = [tree.predict(X) for tree in self.trees]

        # Compute the response
        response = []
        for i in range(len(X)):
            sum_so_far = 0
            for j in range(self.n_trees):
                sum_so_far += predictions[j][i]
            response.append(sum_so_far / self.n_trees)

        return response


def RSS(response):

    # Obtain the main response
    mean_response = response.mean()

    # Compute the RSS
    return sum((response - mean_response)**2), mean_response


def RMSE(y, y_hat):
    return sum((y - y_hat)**2)**(1/2)


if __name__ == "__main__":
    dataset = pd.read_csv("../DecisionTreeRegression/Hitters.csv", index_col="Unnamed: 0")
    dataset = dataset.dropna()

    feature_names = set(dataset.columns) - set(["Salary"]) #["Years", "Hits"]
    features = dataset[feature_names]
    response = dataset["Salary"]
    response = np.log10(response)

    # Create a random forest regressor
    rf_regressor = RandomForestRegressor(n_features=len(feature_names), n_trees=5, min_leaf_size=5, decorrelate_strategy="sqrt")
    rf_regressor.fit(features, response, RSS)

    predictions = rf_regressor.predict(features)

    # Calculate the RMSE
    predictions = np.array(predictions)
    actuals = np.array(response)

    print(RMSE(actuals, predictions))


