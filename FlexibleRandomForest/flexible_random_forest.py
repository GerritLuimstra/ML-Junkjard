"""
    Random Forest support for my custom flexible tree regressor
"""
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import ForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from FlexibleTreeRegressor.flexible_tree_regressor import FlexibleTreeRegressor


class FlexibleRandomForest(ForestRegressor):

    def __init__(self,
                 # Flexible Tree Regressor parameters
                 min_samples_fit=None,
                 leaf_estimator=LinearRegression(),
                 only_path_features=True,

                 n_estimators=100,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 ccp_alpha=0.0,
                 max_samples=None):
        super().__init__(
            base_estimator=FlexibleTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("min_samples_fit", "leaf_estimator", "only_path_features",
                              "criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state", "ccp_alpha"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha
        self.min_samples_fit = min_samples_fit
        self.leaf_estimator = leaf_estimator
        self.only_path_features = only_path_features


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
    hitters_X_train, hitters_X_test, hitters_y_train, hitters_y_test = train_test_split(X_hitters, y_hitters,
                                                                                        test_size=0.33)
    synthetic_X_train, synthetic_X_test, synthetic_y_train, synthetic_y_test = train_test_split(X_synthetic,
                                                                                                y_synthetic,
                                                                                                test_size=0.33)

    # Define the base model
    hitters_base_model = RandomForestRegressor()
    synthetic_base_model = RandomForestRegressor()

    # Train the base models on both training sets
    hitters_base_model.fit(hitters_X_train, hitters_y_train)
    synthetic_base_model.fit(synthetic_X_train, synthetic_y_train)

    # Define the flexible models
    hitters_flex_model = FlexibleRandomForest(min_samples_leaf=10, min_samples_fit=20, only_path_features=False)
    synthetic_flex_model = FlexibleRandomForest(min_samples_leaf=30, min_samples_fit=40, only_path_features=False)

    # To show that it also easily works with other leaf estimators
    hitters_knn_flex_model = FlexibleRandomForest(leaf_estimator=KNeighborsRegressor(), min_samples_leaf=20)

    # Train the flex models
    hitters_flex_model.fit(hitters_X_train, hitters_y_train)
    hitters_knn_flex_model.fit(hitters_X_train, hitters_y_train)
    synthetic_flex_model.fit(synthetic_X_train, synthetic_y_train)

    # Print out the results
    print("Hitters RMSE (BASE RF)")
    print(mean_squared_error(hitters_y_test, hitters_base_model.predict(hitters_X_test)))
    print("Hitters RMSE (FLEX RF )")
    print(mean_squared_error(hitters_y_test, hitters_flex_model.predict(hitters_X_test)))
    print("Hitters RMSE (KNN FLEX RF)")
    print(mean_squared_error(hitters_y_test, hitters_knn_flex_model.predict(hitters_X_test)))
    print("Synthetic RMSE (BASE RF)")
    print(mean_squared_error(synthetic_y_test, synthetic_base_model.predict(synthetic_X_test)))
    print("Synthetic RMSE (FLEX RF)")
    print(mean_squared_error(synthetic_y_test, synthetic_flex_model.predict(synthetic_X_test)))

    # To show that the model works with regular sklearn functionality
    print("Cross validated synthetic score")
    print(-cross_val_score(synthetic_flex_model, synthetic_X_train, synthetic_y_train, cv=10,
                           scoring='neg_mean_squared_error').mean())

    # We see that this model outperforms the base model on the synthetic dataset and does not under perform very bad on the other data

