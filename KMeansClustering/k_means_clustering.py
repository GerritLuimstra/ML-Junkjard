import pandas as pd
from sklearn import datasets
import numpy as np
import numbers
import random
import seaborn as sns
import matplotlib.pyplot as plt


class KMeansClustering:

    def __init__(self, n_centroids, n_iterations):
        self.n_centroids = n_centroids
        self.n_iterators = n_iterations
        self.centroids = []

    """
        Within cluster variation defined in terms of squared Euclidean distance
    """
    def _within_cluster_variation(self, cluster):

        features = cluster.columns
        pair_wise_distances = 0

        # Compute all pair-wise distances
        for _, row_a in cluster.iterrows():
            for _, row_b in cluster.iterrows():
                for p in features:

                    # WCV is only applicable to number content
                    if not isinstance(row_a[p], numbers.Number):
                        continue

                    # Compute the pair_wise distance and add it
                    pair_wise_distances += (row_a[p] - row_b[p])**2

        # We should not forget to divide it by its length
        return pair_wise_distances / len(cluster)

    """
        Obtain all the features of the input that have numeric content
    """
    def _numeric_features(self, X):
        indices = []
        for index, p in enumerate(X.columns):
            if isinstance(X[p].iloc[0], numbers.Number) and p is not "__CLASS__":
                indices.append(index)
        return indices

    """
        Compute the most optimal centroid (the mean of each column in a cluster)
    """
    def _compute_centroid(self, cluster):

        # For each feature of the given cluster, compute the mean
        centroid = []
        for index in self._numeric_features(cluster):
            feature = cluster.iloc[:, index]
            centroid.append(feature.mean())

        return centroid

    """
        Generate a random set of clusters to start with
        In order words, assigns each observation a class such that each cluster has at least one element
    """
    def _generate_random_clusters(self, X):
        classes = np.array([])
        while len(set(classes)) < self.n_centroids:
            classes = np.array([random.randint(1, self.n_centroids) for i in range(len(X))])
        return classes

    """
        Given a data frame A and a centroid B, compute their euclidean distance
    """
    def _euclidean_distance(self, A, B, features):
        distance = 0
        for index, p in enumerate(features):
            distance += (A[p] - B[index])**2
        return distance


    """
        Given input X, compute the centroids that best minimize the within cluster variation
    """
    def fit(self, X):

        # Initialize the centroids
        best_centroids = None
        best_objective_value_so_far = np.inf

        for _ in range(self.n_iterators):

            # Initialize the centroids
            centroids = [[None]] * self.n_centroids

            # Randomly assign each observation a number from 1 - n_centroids
            X["__CLASS__"] = self._generate_random_clusters(X)

            while True:

                # keep track of the new centroids, to know when to stop
                new_centroids = []

                # Each of the clusters, compute the cluster centroid
                for index in range(self.n_centroids):

                    # Obtain all the rows associated with the cluster
                    cluster_observations = X[X["__CLASS__"] == index + 1]

                    # Compute the cluster centroid
                    new_cluster_centroid = self._compute_centroid(cluster_observations)

                    # Find the observation closest to it
                    closest_observation_index = np.argmin(
                        [self._euclidean_distance(observation, new_cluster_centroid, self._numeric_features(X))
                         for _, observation in cluster_observations.iterrows()]
                    )

                    # Obtain the closest observation
                    closest_observation = list(cluster_observations.iloc[closest_observation_index].drop("__CLASS__"))

                    # Add the cluster centroid to the new centroids
                    new_centroids.append(closest_observation)

                # We should stop if the centroids stop changing
                if new_centroids == centroids:
                    break

                # The centroids changed, so let's update them
                centroids = new_centroids

                # Assign each observation to the closest cluster
                for _, observation in X.iterrows():
                    closest_cluster_index = np.argmin([self._euclidean_distance(observation, centroid, self._numeric_features(X)) for centroid in centroids])
                    observation["__CLASS__"] = closest_cluster_index + 1

            # Compute the object value (the sum of the within cluster variation)
            objective_value = 0
            for index in range(self.n_centroids):
                # Obtain all the rows associated with the cluster
                cluster_observations = X[X["__CLASS__"] == index + 1]
                objective_value += self._within_cluster_variation(cluster_observations)

            if objective_value < best_objective_value_so_far:
                best_centroids = centroids
                best_objective_value_so_far = objective_value

        self.centroids = best_centroids
        X.drop(["__CLASS__"], inplace=True, axis=1)

    """
        For a given feature set X, compute the associated class
    """
    def predict(self, X):
        classes = []

        # Assign each observation to the closest cluster
        for _, observation in X.iterrows():
            closest_cluster_index = np.argmin(
                [self._euclidean_distance(observation, centroid, self._numeric_features(X)) for centroid in self.centroids])
            classes.append(closest_cluster_index + 1)

        return classes


if __name__ == "__main__":

    # Load in the IRIS dataset
    dataset = datasets.load_iris()
    dataset = pd.DataFrame(data=np.c_[dataset['data'], dataset['target']],
                           columns=dataset['feature_names'] + ['target'])

    # We don't need the target for now
    dataset = dataset.drop(["target"], axis=1)
    dataset = dataset[["petal width (cm)", "petal length (cm)"]]

    # Plot the data
    sns.set()
    sns.scatterplot(x="petal width (cm)", y="petal length (cm)", data=dataset)
    plt.show()

    cluster = KMeansClustering(n_centroids=2, n_iterations=10)
    cluster.fit(dataset)

    dataset["label"] = cluster.predict(dataset)

    # Plot the data again
    sns.set()
    sns.scatterplot(x="petal width (cm)", y="petal length (cm)", style="label", hue="label", data=dataset)
    plt.show()
