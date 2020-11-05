from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import scipy.spatial.distance
import scipy.cluster.hierarchy


class AgglomerativeClustering:

    def __init__(self, linkage_type, threshold):
        """
        Initializes the agglomerative hierarchical clustering algorithm
        :param linkage_type: The type of linkage to use (must be single, complete, average or ward)
        :param threshold: The threshold to use for the dendrogram to obtain the clusters
        """

        self.linkage_type = linkage_type.lower()
        self.threshold = threshold
        self.labels_ = None
        self.Z = None

        assert linkage_type in ['single', 'complete', 'average', 'ward']

    def fit(self, X):
        """
        Performs the clustering on the given data
        :param X: The data to perform clustering on
        """

        # Compute the pair wise distance matrix of the sample
        Y = scipy.spatial.distance.pdist(X)

        # Compute the linkage scores for each sample
        self.Z = scipy.cluster.hierarchy.linkage(Y, self.linkage_type)

        # Cluster the sample based on the linkage and threshold
        self.labels_ = scipy.cluster.hierarchy.fcluster(self.Z, t=self.threshold, criterion='distance')

    def plot_dendrogram(self):
        """
        Plots the dendrogram from the obtained linkage information
        """
        if self.Z is None:
            raise Exception("Model is not fit!")

        # Visualize the dendrogram
        scipy.cluster.hierarchy.dendrogram(self.Z)
        plt.show()


if __name__ == "__main__":

    # Create a dataset
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42, cluster_std=2)

    # Visualize the dataset
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    # Run the model
    model = AgglomerativeClustering('ward', 40)
    model.fit(X)

    # Plot the dendrogram
    model.plot_dendrogram()

    # Visualize the dataset with the predicted labels
    plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
    plt.show()
