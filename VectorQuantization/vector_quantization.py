import random
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


class VectorQuantization:

    def __init__(self, num_prototypes, learning_rate, max_epochs, seed=42):
        """
        Initializes the Vector Quantization algorithm
        :param num_prototypes: The number of prototypes to use
        :param learning_rate: The learning rate of the prototypes
        :param max_epochs: The maximum amount of iterations to run for
        :param seed: The seed to use for the randomization
        """

        self.num_prototypes = num_prototypes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.seed = seed

        self.prototypes_ = None
        self.labels_ = None

    def fit(self, X):
        """
        Performs the VQ clustering on the given data
        :param X: The given data
        """

        # Make a copy, so we do not change the data
        X_ = np.copy(X)

        # Initialize the prototypes to be a sample vector, so we are already on the data manifold
        prototypes = random.choices(X, k=self.num_prototypes)

        for epoch in range(self.max_epochs):

            # # Shuffle the data, to prevent order artifacts
            np.random.shuffle(X_)

            for index, sample in enumerate(X_):

                # Find the prototype closest to the data point
                closest = np.argmin([distance_function(sample, prototype) for prototype in prototypes])

                # Move it closer to the data point
                prototypes[closest] += self.learning_rate * (sample - prototypes[closest])

        # Assign the labels to the given data, based on the prototype vectors
        self.labels_ = [np.argmin([distance_function(sample, prototype)
                        for prototype in prototypes]) for sample in X]

        self.prototypes_ = np.array(prototypes)


if __name__ == "__main__":

    # Create a dataset
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42, cluster_std=2)

    # Visualize the dataset
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    # Run the model
    model = VectorQuantization(3, 0.001, 100)
    model.fit(X)

    # Visualize the dataset
    plt.scatter(X[:, 0], X[:, 1], c=model.labels_, label="Data")
    plt.scatter(model.prototypes_[:, 0], model.prototypes_[:, 1], label="Prototypes",
                c=list(range(len(model.prototypes_))),
                marker="*", s=500)
    plt.legend()
    plt.show()  # Looks good!
