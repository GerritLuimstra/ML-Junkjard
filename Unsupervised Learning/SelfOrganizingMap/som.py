from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import random


def euclidean_distance(p, q):
    """
    Simple euclidean distance
    :param p: The first vector
    :param q: The second vector
    :return: The euclidean distance between both vectors
    """
    return np.linalg.norm(p - q)


def manhattan_distance(v1, v2):
    """
    Computes the manhattan distance between two vectors
    :param v1: The first vector
    :param v2: The second vector
    """
    return sum(abs(p - q) for p, q in zip(v1, v2))


class KohonenSOM:

    def __init__(self, map_size, epochs):
        """
        Initializes the Kohonen Self Organizing Map

        :param map_size: The size of the node map. Note, the amount of nodes is the square of this number.
        :param epochs: The amount of epochs to train for

        NOTE: In this (simplified) implementation, we have a linearly decaying square size and learning rate
        """
        self.map_size = map_size
        self.epochs = epochs
        self.nodes_ = []
        self.labels_ = None

    def fit(self, X: np.array):
        """
        Performs the SOM algorithm on the given data
        :param X: The given data
        """

        # Randomly initialize the nodes with an input sample
        kohonen_map = []
        for x in range(self.map_size):
            for y in range(self.map_size):
                kohonen_map.append((x, y, random.choice(X)))

        for epoch in range(self.epochs):

            # Calculate the square size (note: this is a simple linear decay)
            square_size = (self.map_size / 2) * (1 - epoch / self.epochs)

            # Calculate the learning rate (note: again a simple linear decay)
            learning_rate = 0.8 * (1 - epoch / self.epochs)

            # Loop over the sample, find the BMU and update its neighbourhood to be more similar to the input sample
            for sample in X:

                # Find the distance to all nodes
                distances = [euclidean_distance(sample, node[2]) for node in kohonen_map]

                # Find the best matching unit
                best_matching_unit = kohonen_map[np.argmin(distances)]

                # Find the neighbour hood of the best matching unit
                neighbourhood = [index for index, cluster in enumerate(kohonen_map)
                                 if manhattan_distance(best_matching_unit[:2], cluster[:2]) <= square_size]

                # Update each neighbour in the neighbourhood
                for neighbour in neighbourhood:

                    # Obtain the updated neighbour
                    updated_neighbour = (1 - learning_rate) * kohonen_map[neighbour][2] + learning_rate * sample

                    # Actually update the neighbour
                    kohonen_map[neighbour] = (kohonen_map[neighbour][0], kohonen_map[neighbour][1], updated_neighbour)

        # Assign each sample to its cluster
        self.labels_ = [np.argmin([euclidean_distance(sample, node[2])
                                   for node in kohonen_map]) for sample in X]
        # Extract the node vectors
        self.nodes_ = np.array([node[2] for node in kohonen_map])


if __name__ == "__main__":

    # Create a dataset
    X, y = make_blobs(n_samples=500, centers=15, n_features=2, random_state=42, cluster_std=1.3)

    # # Visualize the dataset
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    # Run the model
    model = KohonenSOM(3, 100)
    model.fit(X)

    # Visualize the dataset
    unique_labels = list(set(model.labels_))
    labels = [unique_labels.index(label) for label in model.labels_]
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(model.nodes_[:, 0], model.nodes_[:, 1], label="Prototypes",
                c=list(range(len(model.nodes_))),
                marker="*", s=500)
    plt.show()  # Looks good!
