"""
    The following model is a custom model that I created in order to have more flexibility over the built-in decision tree regressor of sklearn.
    By default a the scikit-learn decision tree regressor predicts a constant value in the leave nodes. However,
    rather than a simple constant, perhaps it is nice to be able to fit a model on such leaf node data.
    Loosely modelled after Quinlan's M5 trees, I have implemented something I coin a "Flexible Tree Regressor".

    Note: This model extends rather than changes the functionality of the base regressor of sklearn and
          due to this it is still compatible with all other tooling surrounding these models.
"""
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import copy
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor


class FlexibleTreeRegressor(DecisionTreeRegressor):

    def __init__(self,
                 # Custom FlexibleTreeRegressor settings
                 min_samples_fit=None,
                 leaf_estimator=LinearRegression(),
                 only_path_features=True,

                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 presort='deprecated',
                 ccp_alpha=0.0):
        """
        Initialization for the Flexible Tree Regressor

        :param min_samples_fit : Only fit a model if the leaf contains this much samples
        :param leaf_estimator: The estimator to use on the leaf nodes
        :param only_path_features: Whether to only restrict the leaf-models to the features of the path
                                   This makes it behave similarly to an M5 model

        Note: The other parameters are for sklearn itself
        """
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            presort=presort,
            ccp_alpha=ccp_alpha)

        # Setup the required properties
        self._node_lookup = {}
        self.min_samples_fit = min_samples_fit
        self.leaf_estimator = leaf_estimator
        self.only_path_features = only_path_features

    def fit(self, X, y, *args, **kwargs):
        """
        Fits the Flexible Tree Regressor to the given training data

        :param X: The training features
        :param y: The training response
        :param args: Additional positional arguments passed
        :param kwargs: Additional keyword arguments passed
        """

        # Call the fit function on the base decision tree regressor
        super().fit(X, y, *args, **kwargs)

        # For each sample in the dataset, obtain in which node they belong
        nodes = super().apply(X)

        # Build a lookup table for each node
        for unique_node in np.unique(nodes):
            self._node_lookup[unique_node] = {"node_features_in_path": set([]), "samples": [], "response": [], "model": None}

        # Since sklearn (for good reasons) does not store which sample belongs in what node,
        # we need to similary this behaviour by keeping track of which samples go into what node
        for index, node in enumerate(nodes):
            self._node_lookup[node]["samples"].append(X.iloc[index])
            self._node_lookup[node]["response"].append(y.iloc[index])

        # If the user wishes to only use the features using the path up for each node,
        # we need to obtain those features (sklearn does not provide them by default)
        if self.only_path_features:

            # For each leave, obtain the features used
            for node in self._node_lookup:

                # If set, skip leafs that have less than min_samples_fit samples
                if self.min_samples_fit is not None:
                    if len(self._node_lookup[node]["samples"]) < self.min_samples_fit:
                        continue

                # Obtain a sample
                sample = self._node_lookup[node]["samples"][0]

                # Obtain the path
                path = self.decision_path([sample]).toarray()[0]

                # Obtain the feature names
                indices = np.nonzero(path)
                _features = self.tree_.feature[indices]
                _features = np.unique(_features[_features >= 0])

                # Check to see if sklearn was able to make a split with these settings
                if _features.size == 0:
                    raise RuntimeError("The base sklearn tree could not be made. Try different settings.")

                # Add the features used to the path of this node
                self._node_lookup[node]["node_features_in_path"] = set(X.columns[_features])

        # Fit the selected model to the samples in each respective leaf node
        for node in self._node_lookup:

            node_samples = self._node_lookup[node]["samples"]
            features_in_path = self._node_lookup[node]["node_features_in_path"]

            # If set, skip leafs that have less than min_samples_fit samples
            if self.min_samples_fit is not None:
                if len(node_samples) < self.min_samples_fit:
                    continue

            # If the user so wishes, only use the features in the path down
            if self.only_path_features:
                for index, sample in enumerate(node_samples):
                    self._node_lookup[node]["samples"][index] = sample[list(features_in_path)]

            # Fit the model to the samples in the leaf node
            self._node_lookup[node]["model"] = copy.copy(self.leaf_estimator).fit(node_samples, self._node_lookup[node]["response"])

            # Remove the samples and responses (free memory)
            self._node_lookup[node]["samples"] = None
            self._node_lookup[node]["response"] = None

    def predict(self, X, *args, **kwargs):
        """
        Obtains the response for the given data
        :param X: The data to be prediction on
        :param args: Additional positional arguments passed
        :param kwargs: Additional keyword arguments passed
        :return: The predicted response variable
        """
        predictions = []

        for index, node in enumerate(self.apply(X)):

            # Predict the normal value, if there is no model fit
            # This is due to the the user setting a min_samples_split
            if self._node_lookup[node]["model"] is None:
                predictions.append(super().predict([X.iloc[index]])[0])
                continue

            # Obtain the sample to be predicted
            sample = X.iloc[index]

            # If set, only use the features that are along the path to the associated leaf
            if self.only_path_features:
                sample = sample[list(self._node_lookup[node]["node_features_in_path"])]

            # Make the prediction
            predictions.append(self._node_lookup[node]["model"].predict([np.array(sample)])[0])

        return np.array(predictions)


if __name__ == "__main__":

    # Load in the hitters dataset
    hitters_dataset = pd.read_csv("../DecisionTreeRegression/Hitters.csv", index_col="Unnamed: 0")
    hitters_dataset = hitters_dataset.dropna()
    feature_names = set(hitters_dataset.columns) - {"Salary", "NewLeague", "Division", "League"}
    features = hitters_dataset[feature_names]
    response = hitters_dataset["Salary"]
    response = np.log10(response)
    X_hitters = features
    y_hitters = response

    # Generate a synthetic dataset
    X_, y_, coef = datasets.make_regression(n_samples=300, n_features=4,
                                            n_informative=1, noise=10,
                                            coef=True, random_state=0)

    dataset = pd.DataFrame({"X1": X_[:, 0], "X2": X_[:, 1], "X3": X_[:, 2], "X4": X_[:, 3], "y": y_})
    X_synthetic = dataset[["X1", "X2", "X3", "X4"]]
    y_synthetic = dataset["y"]

    # Create a train and test set
    hitters_X_train, hitters_X_test, hitters_y_train, hitters_y_test = train_test_split(X_hitters, y_hitters, test_size=0.33)
    synthetic_X_train, synthetic_X_test, synthetic_y_train, synthetic_y_test = train_test_split(X_synthetic, y_synthetic, test_size=0.33)

    # Define the base model
    hitters_base_model = DecisionTreeRegressor()
    synthetic_base_model = DecisionTreeRegressor()

    # Train the base models on both training sets
    hitters_base_model.fit(hitters_X_train, hitters_y_train)
    synthetic_base_model.fit(synthetic_X_train, synthetic_y_train)

    # Define the flexible model
    hitters_flex_model = FlexibleTreeRegressor(min_samples_leaf=20, min_samples_fit=30, only_path_features=False)
    synthetic_flex_model = FlexibleTreeRegressor(min_samples_leaf=30, min_samples_fit=15, only_path_features=False)

    # To show that it also easily works with other leaf estimators
    hitters_knn_flex_model = FlexibleTreeRegressor(leaf_estimator=KNeighborsRegressor(), min_samples_leaf=20)

    # Train the flex models
    hitters_flex_model.fit(hitters_X_train, hitters_y_train)
    hitters_knn_flex_model.fit(hitters_X_train, hitters_y_train)
    synthetic_flex_model.fit(synthetic_X_train, synthetic_y_train)

    # Print out the results
    print("Hitters RMSE (BASE)")
    print(mean_squared_error(hitters_y_test, hitters_base_model.predict(hitters_X_test)))
    print("Hitters RMSE (FLEX)")
    print(mean_squared_error(hitters_y_test, hitters_flex_model.predict(hitters_X_test)))
    print("Hitters RMSE (KNN FLEX)")
    print(mean_squared_error(hitters_y_test, hitters_knn_flex_model.predict(hitters_X_test)))
    print("Synthetic RMSE (BASE)")
    print(mean_squared_error(synthetic_y_test, synthetic_base_model.predict(synthetic_X_test)))
    print("Synthetic RMSE (FLEX)")
    print(mean_squared_error(synthetic_y_test, synthetic_flex_model.predict(synthetic_X_test)))

    # To show that the model works with regular sklearn functionality
    print("Cross validated synthetic score")
    print(-cross_val_score(synthetic_flex_model, synthetic_X_train, synthetic_y_train, cv=10, scoring='neg_mean_squared_error').mean())

    # We see that this model outperforms the base model (on these two datasets)

















