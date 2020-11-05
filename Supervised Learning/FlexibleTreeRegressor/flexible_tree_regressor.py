"""
    The following model is a custom model that I created in order to have more flexibility over the built-in decision tree regressor of sklearn.
    By default a the scikit-learn decision tree regressor predicts a constant value in the leave nodes. However,
    rather than a simple constant, perhaps it is nice to be able to fit a model on such leaf node data.
    Loosely modelled after Quinlan's M5 trees, I have implemented something I coin a "Flexible Tree Regressor".

    Optionally, a custom 'clipping trick' can be used in which the model tries to determine in which cases it is smart to use the leaf model.

    Note: This model extends rather than changes the functionality of the base regressor of sklearn and
          due to this it is still compatible with all other tooling surrounding these models.
"""
from sklearn.linear_model import LinearRegression
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
                 only_path_features=False,
                 use_clipping_trick=True,

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
        :param use_clipping_trick: Whether or not to use the clipping trick

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
        self.use_clipping_trick = use_clipping_trick

    def fit(self, X, y, *args, **kwargs):
        """
        Fits the Flexible Tree Regressor to the given training data

        :param X: The training features
        :param y: The training response
        :param args: Additional positional arguments passed
        :param kwargs: Additional keyword arguments passed
        """

        X = np.array(X)
        y = np.array(y)

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

            if not type(X) == np.ndarray:
                self._node_lookup[node]["samples"].append(np.array(X.iloc[index]))
                self._node_lookup[node]["response"].append(np.array(y.iloc[index]))
            else:
                self._node_lookup[node]["samples"].append(X[index])
                self._node_lookup[node]["response"].append(y[index])

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
                self._node_lookup[node]["node_features_in_path"] = set(_features)

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

            # Compute the cluster centroid
            self._node_lookup[node]["centroid"] = np.hstack((np.mean(np.array(self._node_lookup[node]["samples"]), axis=0),
                                                   np.mean(self._node_lookup[node]["response"])))

            # Convert the data to numpy arrays to be dealt with later
            self._node_lookup[node]["samples"] = None
            self._node_lookup[node]["response"] = None

    def _distance(self, v1, v2, p):
        """
        Computes the distance between two vectors

        :param np.array v1: The first vector
        :param np.array v2: The second vector
        :param int p : The 1/p exponent is used to define the distance metric (based on the Minkowski formula)
        """
        return sum(abs(v1 - v2)**p)**(1/p)

    def _clip_trick(self, sample, node):
        """
        Finds the optimal prediction between the mean vector (the default outcome) or the predicted point on the line

        It determines whether the point (X, 0) is closer to (X_mu, y_mu) or (X, y_pred).
        It will choose the optimal prediction based on this insight.

        :param sample: The sample to predict
        :param node: The node the sample is in
        :return: The optimal outcome of either the mean or the predicted point by the regression line
        """

        # Compute all the necessary information for the clipping trick
        sample_point = np.hstack((np.array(sample), [0]))
        mean_point = self._node_lookup[node]["centroid"]
        prediction = self._node_lookup[node]["model"].predict([sample])[0]

        if type(prediction) == np.ndarray:
            prediction = prediction[0]

        prediction_point = np.hstack((np.array(sample), [prediction]))

        # Compute the distances between the sample point and mean vs the prediction point
        prediction_distance = self._distance(sample_point, prediction_point, 2)
        mean_distance = self._distance(sample_point, mean_point, 2)

        # See which prediction is closer
        return prediction if prediction_distance < mean_distance else mean_point[-1]

    def predict(self, X, *args, **kwargs):
        """
        Obtains the response for the given data
        :param X: The data to be prediction on
        :param args: Additional positional arguments passed
        :param kwargs: Additional keyword arguments passed
        :return: The predicted response variable
        """
        X = np.array(X)

        predictions = []

        for index, node in enumerate(self.apply(X)):

            # Predict the normal value, if there is no model fit
            # This is due to the the user setting a min_samples_split
            if self._node_lookup[node]["model"] is None:
                predictions.append(super().predict([X[index]])[0])
                continue

            # Obtain the sample to be predicted
            sample = X[index]

            # If set, only use the features that are along the path to the associated leaf
            if self.only_path_features:
                sample = sample[list(self._node_lookup[node]["node_features_in_path"])]

            # See if we can use the clipping trick
            if isinstance(self._node_lookup[node]["model"], LinearRegression) and self.use_clipping_trick:
                # Make the prediction using the clip trick
                predictions.append(self._clip_trick(sample, node))
            else:
                # Make a regular prediction
                predictions.append(self._node_lookup[node]["model"].predict([np.array(sample)])[0])

        return np.array(predictions, dtype=np.float64).reshape((X.shape[0]))


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
    hitters_flex_model = FlexibleTreeRegressor(min_samples_leaf=20, min_samples_fit=30, only_path_features=False, use_clipping_trick=True)
    synthetic_flex_model = FlexibleTreeRegressor(min_samples_leaf=20, min_samples_fit=30, only_path_features=False, use_clipping_trick=True)

    # To show that it also easily works with other leaf estimators
    hitters_knn_flex_model = FlexibleTreeRegressor(leaf_estimator=KNeighborsRegressor(), min_samples_leaf=20)

    # Train the flex models
    hitters_flex_model.fit(hitters_X_train, hitters_y_train)
    hitters_knn_flex_model.fit(hitters_X_train, hitters_y_train)
    synthetic_flex_model.fit(synthetic_X_train, synthetic_y_train)

    # Print out the results
    print("Hitters RMSE (BASE)")
    print(mean_squared_error(hitters_y_test, hitters_base_model.predict(hitters_X_test)))
    print("Hitters RMSE (BASE CV)")
    print(-cross_val_score(hitters_base_model, hitters_X_train, hitters_y_train, cv=10, scoring='neg_mean_squared_error').mean())
    print("\nHitters RMSE (FLEX)")
    print(mean_squared_error(hitters_y_test, hitters_flex_model.predict(hitters_X_test)))
    print("Hitters RMSE (FLEX CV)")
    print(-cross_val_score(hitters_flex_model, hitters_X_train, hitters_y_train, cv=10, scoring='neg_mean_squared_error').mean())
    print("\nHitters RMSE (KNN FLEX)")
    print(mean_squared_error(hitters_y_test, hitters_knn_flex_model.predict(hitters_X_test)))
    print("Hitters RMSE (KNN FLEX CV)")
    print(-cross_val_score(hitters_knn_flex_model, hitters_X_train, hitters_y_train, cv=10, scoring='neg_mean_squared_error').mean())

    print("\nSynthetic RMSE (BASE)")
    print(mean_squared_error(synthetic_y_test, synthetic_base_model.predict(synthetic_X_test)))
    print("Synthetic RMSE (FLEX)")
    print(mean_squared_error(synthetic_y_test, synthetic_flex_model.predict(synthetic_X_test)))

    # To show that the model works with regular sklearn functionality
    print("\nCross validated synthetic score")
    print(-cross_val_score(hitters_flex_model, synthetic_X_train, synthetic_y_train, cv=10, scoring='neg_mean_squared_error').mean())

    # We see that this model outperforms the base model (on these two datasets)

















