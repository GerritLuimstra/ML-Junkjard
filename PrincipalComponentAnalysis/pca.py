"""
    PCA implementation

    Resource: https://www.youtube.com/watch?v=ZqXnPcyIAL8
"""
import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA as PCA_SKLEARN
from sklearn import preprocessing


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.component_variance_explained = None
        self.total_variance_explained = None
        self.W = None

    def fit(self, X):
        """
        Obtains the first 'k' principal components of the given data
        NOTE: This does not apply any centering/scaling!

        :param pandas.DataFrame X: The data of which the components will be extracted
        """

        if self.n_components > X.shape[1]:
            raise Exception("ERROR: Attempt to use more components than features!")

        # Scale the input
        X_scaled = (X - np.mean(X, axis=0))

        # Compute the covariance matrix
        C = np.cov(X_scaled.T)

        # Compute the eigen-values and vectors of the covariance matrix
        eigen_values, eigen_vectors = LA.eig(C)  # TODO: we should probably use eigh here!

        # Sort the eigen vectors by their eigen values
        sorted_indices = np.argsort(eigen_values)[::-1]
        sorted_eigen_vectors = eigen_vectors[:, sorted_indices]

        # Compute the variance explained
        total_variance = sum(eigen_values)
        self.component_variance_explained = [(eigen_value / total_variance)
                                             for eigen_value in sorted(eigen_values, reverse=True)[:self.n_components]]
        self.total_variance_explained = sum(self.component_variance_explained[:self.n_components])

        # Select the first 'k' eigen vectors to form W
        self.W = sorted_eigen_vectors[:, :self.n_components]

    def transform(self, X):
        """
        Transforms the given input X using the principal components obtained from the fit
        NOTE: This does not apply any centering/scaling!

        :param pandas.DataFrame X: The data to be transformed
        """

        if self.W is None:
            raise Exception("PrincipalComponentAnalysis is not fit yet!")

        # Scale the input
        X_scaled = (X - np.mean(X, axis=0))

        return np.dot(X_scaled, self.W)


if __name__ == "__main__":

    X1 = [90, 90, 60, 60, 30]
    X2 = [60, 90, 60, 60, 30]
    X3 = [90, 30, 60, 90, 30]

    X = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})

    # Fit our own PCA to the data
    pca = PCA(n_components=2)
    pca.fit(X)

    # Fit the scikit learn version of PCA to the data
    pca_sklearn = PCA_SKLEARN(n_components=2)
    pca_sklearn.fit(X)

    # Verify that they are equal
    print(pca.component_variance_explained)
    print(pca_sklearn.explained_variance_ratio_)

    # They are equal up to a sign change
    # See: https://stackoverflow.com/questions/58666635/implementing-pca-with-numpy
    print(pca.transform(X))
    print(pca_sklearn.transform(X))
