"""
    Please note that I could not easily find the inner workings of this rather old algorithm (Quinlan 1992).
    Hence, I implemented it in the way I thought was intended.

    In my view, an M5 Regression tree builds up a plain regression tree.
    Next, rather than predicting a constant value in the leaves, it uses the predictors from the path down
    to build a simple linear regression model on the data points from that leave.
    To prevent overfitting, the minimum amount of leaves in a split should be increased.

    NOTE: I could not figure out what the criterion value was for this algorithm, so I left it unmodified.
          Luckily, the M5 model did seem to work better than the regular decision tree model.
"""

from DecisionTreeRegression.regression_tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd
from util.criterions import RSS


class M5DecisionTreeRegressor:

    def __init__(self, min_leaf_size=2, max_splits=None):
        self.min_leaf_size = min_leaf_size
        self.feature_names = None
        self.max_splits = max_splits
        self.current_splits = 0

        # We start with an empty decision tree regressor
        self.tree = DecisionTreeRegressor()

        # Overwrite the predict function
        self.tree._predict = self._predict

    """
        Fits a decision tree based on the features and responses
    """
    def fit(self, features, response, criterion_function):

        # Fit the decision tree regressor to the data
        self.tree.fit(features, response, criterion_function)

        # Loop over the tree and train the linear models on the leaves
        self._train_linear_models(self.tree.tree, features, response, [])

    """
        Given an input X, compute the response Y
    """
    def predict(self, X):
        return self.tree.predict(X)

    """
        Trains linear models on the leaves of the decision tree using only the features from its path (to the root)
    """
    def _train_linear_models(self, tree, features, response, path):

        # We have reached a leave
        if tree.split_point is None or len(features) < self.min_leaf_size:

            # Train a linear model using only the features from the path up (to the root)
            tree.model = LinearRegression().fit(features[list(set(path))], response)

            # Set the features used
            tree.model_features = path

            return

        # Obtain the split point
        split_value, feature_name, is_number, _ = tree.split_point

        if is_number:

            # Prepare the left and right features and responses
            left_features = features[features[feature_name] < split_value]
            left_response = response[features[feature_name] < split_value]

            # Prepare the left and right features and responses
            right_features = features[features[feature_name] >= split_value]
            right_response = response[features[feature_name] >= split_value]

        else:

            # Prepare the left and right features and responses
            left_features = features[features[feature_name] == split_value]
            left_response = response[features[feature_name] == split_value]

            # Prepare the left and right features and responses
            right_features = features[features[feature_name] != split_value]
            right_response = response[features[feature_name] != split_value]

        # Recursively go over the tree, until we hit a node
        self._train_linear_models(tree.left, left_features, left_response, path + [feature_name])
        self._train_linear_models(tree.right, right_features, right_response, path + [feature_name])

    """
        Given a row, compute the response
    """
    def _predict(self, row):

        # Start at the top of the tree
        tree = self.tree.tree

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
        # Now see if we can use a model
        if tree.model is not None:
            return tree.model.predict([row[list(set(tree.model_features))]])[0]

        # There was no model available, simply use the mean response
        return tree.response


if __name__ == "__main__":

    # Make a simple regression dataset
    X, y, coef = datasets.make_regression(n_samples=300, n_features=4,
                                          n_informative=1, noise=10,
                                          coef=True, random_state=0)

    # Wrangle the dataset
    dataset = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "X3": X[:, 2], "X4": X[:, 3], "y": y})
    X = dataset[["X1", "X2", "X3", "X4"]]
    y = dataset["y"]

    # Create a train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Fit a base model
    model = DecisionTreeRegressor(min_leaf_size=10)
    model.fit(X_train, y_train, RSS)

    print("BASE RMSE (TRAIN)")
    print(mean_squared_error(y_train, model.predict(X_train)))
    print("BASE RMSE (TEST)")
    print(mean_squared_error(y_test, model.predict(X_test)))

    # Fit the M5 model
    m5model = M5DecisionTreeRegressor(min_leaf_size=10)
    m5model.fit(X, y, RSS)

    print("M5 RMSE (TRAIN)")
    print(mean_squared_error(y_train, m5model.predict(X_train)))
    print("M5 RMSE (TEST)")
    print(mean_squared_error(y_test, m5model.predict(X_test)))











