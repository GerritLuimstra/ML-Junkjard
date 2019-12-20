import numbers
import pandas as pd
import numpy as np


class RegressionTree:

    def __init__(self):
        self.split_point = None
        self.left = None
        self.right = None
        self.response = None


class DecisionTreeRegressor:

    def __init__(self, min_leaf_size=2, max_splits=None):
        self.min_leaf_size = min_leaf_size
        self.feature_names = None
        self.max_splits = max_splits
        self.current_splits = 0

        # We start with an empty regression tree
        self.tree = RegressionTree()

    """
        Fits a decision tree based on the features and responses
    """
    def fit(self, features, response, criterion_function):

        # Obtain the feature names
        self.feature_names = features.columns

        # Recursively build up the tree
        self._build_tree(self.tree, features, response, criterion_function)

    """
        Given an input X, compute the response Y
    """
    def predict(self, X):

        # Check if the model has already been fit
        if self.feature_names is None:
            raise Exception("This model has not been fit yet!")

        # Check if we have the right columns
        if not set(self.feature_names).issubset(set(X.columns)):
            raise Exception("Please provide all the required feature names")

        # Compute the responses
        responses = []
        for _, row in X.iterrows():
            responses.append(self._predict(row))

        return responses

    """
        Given a row, compute the response
    """
    def _predict(self, row):

        # Start at the top of the tree
        tree = self.tree

        while tree.split_point is not None:

            # Obtain the split point
            split_value, feature_name, is_number, _ = tree.split_point

            # Determine whether we are dealing with strings or numbers
            if is_number is True:
                if row[feature_name] < split_value:
                    tree = tree.left
                else:
                    tree = tree.right
            else:
                if row[feature_name] == split_value:
                    tree = tree.left
                else:
                    tree = tree.right

        # At this point we are at a terminal node
        # and hence we can simply return the value
        return tree.response

    """
        Builds up the regression tree based on the features, criterion function and responses
    """
    def _build_tree(self, tree, features, response, criterion_function):

        # Find the optimal split point
        optimal_split_point = self._find_optimal_split_point(features, response, criterion_function)

        # If the split cannot be made, return a terminal node
        if optimal_split_point is None:

            # Create the response tree
            response_tree = RegressionTree()
            response_tree.response = response.mean()

            return response_tree

        # Only allow a certain amount of splits, if set
        if self.max_splits is not None and self.current_splits >= self.max_splits:
            return None

        # At this point, a split was made
        split_point, left_branch, left_response, right_branch, right_response = self._find_optimal_split_point(features, response, criterion_function)

        # Keep track of the current splits made
        self.current_splits += 1

        # Set the split point
        tree.split_point = split_point

        # Build up the left and right tree
        tree.left = (self._build_tree(RegressionTree(), left_branch, left_response, criterion_function))
        tree.right = (self._build_tree(RegressionTree(), right_branch, right_response, criterion_function))

        return tree

    """
        Determine whether we are dealing with numbers
    """
    def _isnumber(self, value):
        return isinstance(value, numbers.Number)

    """
        For a given set of features and responses, compute the optimal split point based on the criterion value
    """
    def _find_optimal_split_point(self, features, response, criterion):

        # Check if it is possible to split to start with
        if len(features) < 2 * self.min_leaf_size:
            return None

        optimal_split = (None, None, None, float('inf'))
        best_left_branch = None
        best_right_branch = None

        for feature_name in features.columns:

            for value in set(features[feature_name].values):

                # Obtain the indices of the left and right branch
                if self._isnumber(value):
                    left_branch = features[features[feature_name] < value].index
                    right_branch = features[features[feature_name] >= value].index
                else:
                    left_branch = features[features[feature_name] == value].index
                    right_branch = features[features[feature_name] != value].index

                # Skip the branches that do not have enough leave elements
                if len(left_branch) < self.min_leaf_size or len(right_branch) < self.min_leaf_size:
                    continue

                # Calculate the criterion values for both branches
                left_branch_criterion_value, left_branch_mean_response = criterion(response.loc[left_branch])
                right_branch_criterion_value, right_branch_mean_response = criterion(response.loc[right_branch])

                # Compute the overall criterion value
                criterion_value = left_branch_criterion_value + right_branch_criterion_value

                if criterion_value < optimal_split[3]:
                    optimal_split = (value, feature_name, self._isnumber(value), criterion_value)
                    best_left_branch = left_branch
                    best_right_branch = right_branch

        # Sometimes a split can truly not be made
        if best_left_branch is None or best_right_branch is None:
            return None

        left_response = response.loc[best_left_branch]
        left_branch = features.loc[best_left_branch]
        right_response = response.loc[best_right_branch]
        right_branch = features.loc[best_right_branch]

        return optimal_split, left_branch, left_response, right_branch, right_response


def RSS(response):

    # Obtain the main response
    mean_response = response.mean()

    # Compute the RSS
    return sum((response - mean_response)**2), mean_response


def RMSE(y, y_hat):
    return sum((y - y_hat)**2)**(1/2)


if __name__ == "__main__":

    dataset = pd.read_csv("Hitters.csv", index_col="Unnamed: 0")
    dataset = dataset.dropna()

    feature_names = set(dataset.columns) - set(["Salary"]) #["Years", "Hits"]
    features = dataset[feature_names]
    response = dataset["Salary"]
    response = np.log10(response)

    # Define the regressor
    regressor = DecisionTreeRegressor(min_leaf_size=5)
    regressor.fit(features, response, RSS)
    predictions = regressor.predict(features)

    # Calculate the RMSE
    predictions = np.array(predictions)
    actuals = np.array(response)

    print(RMSE(actuals, predictions))










