from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt


def distance_function(p, q):
    """
    Simple euclidean distance
    :param p: The first vector
    :param q: The second vector
    :return: The euclidean distance between both vectors
    """
    return np.linalg.norm(p - q)


class DBScan:

    def __init__(self, eps, min_samples):
        """
        Initalizes the DBScan clustering algorithm

        :param eps: The maximum distance between two points for one to be considered part of each others neighbourhood
        :param min_samples: The minimum number of samples for a set of points to be considered a cluster
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X: np.array):
        """
        Performs the clustering on the given data
        :param X: The given data
        """

        # Initialize the algorithm with zero clusters
        # and create a set of empty labels
        num_clusters = 0
        labels = [None] * len(X)

        # Loop through all points in the dataset.
        for index, point in enumerate(X):

            # Skip point if it is already labeled.
            if labels[index] != None:
                continue

            # Find the neighbours, label the point noise if there aren't enough neighbours.
            neighbours = self.get_neighbours(X, point, self.eps)
            if len(neighbours) < self.min_samples:
                labels[index] = "Noise"
                continue

            # If there are enough neighbours, give the point a new label and investigate neighbours.
            num_clusters += 1
            labels[index] = num_clusters
            seed = neighbours - set([index])

            # Recursively add neighbours to cluster.
            while len(seed) != 0:

                neighbour = list(seed)[0]

                if labels[neighbour] == "Noise":
                    labels[neighbour] = num_clusters

                if labels[neighbour] != None:
                    seed.remove(neighbour)
                    continue

                labels[neighbour] = num_clusters

                neighbours_ = self.get_neighbours(X, X[neighbour], self.eps)
                if len(neighbours_) >= self.min_samples:
                    seed = seed.union(neighbours_)

        self.labels_ = np.array(labels)

    def get_neighbours(self, X, target, eps):
        """
        :param X: The data to be searched on
        :param target: The sample for which to find the neighbourhood
        :param eps: The maximum distance from the target to a sample point for it to be considered in the neighbourhood
        :return: The set of points in the neighbourhood of the given target
        """
        neighbours = set([])
        for index, point in enumerate(X):
            if distance_function(point, target) <= eps:
                neighbours.add(index)
        return neighbours


if __name__ == "__main__":

    # Create a dataset
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42, cluster_std=2)

    # Visualize the dataset
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    # Run the model
    model = DBScan(1.5, 10)
    model.fit(X)

    # Visualize the dataset
    unique_labels = list(set(model.labels_))
    labels = [unique_labels.index(label) for label in model.labels_]
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show() # Looks good!

