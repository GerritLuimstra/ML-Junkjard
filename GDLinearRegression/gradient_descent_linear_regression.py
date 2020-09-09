import time
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def plot_linear_model(X, y, W):
    """
    Plots the data and the line as given by the vector W
    :param X: The data to be plot on the x-axis
    :param y: The data to be plot on the y-axis
    :param W: The coefficients of the line
    """
    plt.close()
    plt.scatter(X[:, 1], y)
    plt.plot([-3, 3], [-3*W[1] + W[0], 3*W[1] + W[0]])
    plt.show()


class GDLinearRegressor:
    def __init__(self, learning_rate, include_bias=True):
        """
        Initializes the Gradient Descent Linear Regressor
        :param learning_rate: The learning rate with which to converge to the global minimum
        :param include_bias: Whether or not to fit a bias term as well
        """
        self.learning_rate = learning_rate
        self.W = None
        self.include_bias = include_bias
        self.epsilon = 10**(-10)

    @staticmethod
    def gradient(X, y, W):
        """
        Obtains the gradient for a given vector W and data X and response variable y

        The gradient is given by the formula that is found at:
        https://en.wikipedia.org/wiki/Linear_regression#Least-squares_estimation_and_related_techniques

        :param X: The features to be regressed on
        :param y: The response variable
        :param W: The current coefficient matrix
        :return: the gradient
        """
        return - 2 * (y.T.dot(X)) + 2 * W.T.dot(X.T.dot(X))

    def fit(self, X, y):
        """
        Fits a linear regression model using gradient descent on data X based on response variable y
        :param X: The features of the model
        :param y: The response variable
        """

        # Obtain a random set of coefficients
        self.W = np.random.rand(X.shape[1] + int(self.include_bias), 1)

        # If there is a bias, we add a column of ones to the feature vectors
        if self.include_bias:
            X = np.insert(X, 0, values=1, axis=1)

        while True:

            # Obtain the new weight vector
            W_hat = self.W - self.learning_rate * self.gradient(X, y, self.W).T

            # Stop if there is no substantial progress
            # Note that since in machine learning the generalization error is usually
            # more sought after than the true minimum, this is not a problem
            if np.all(np.isclose(W_hat, self.W, atol=self.epsilon)):
                break

            # Update the weight vector
            self.W = W_hat

            # Plot the results
            plot_linear_model(X, y, self.W)
            time.sleep(0.5)

    def predict(self, X):
        """
        Obtains the estimated response to the given samples matrix X

        :param X: The sample matrix
        :return: The response estimates
        """

        # We need a weight vector
        assert self.W is not None

        # If there is a bias, we add a column of ones to the feature vectors
        if self.include_bias:
            X = np.insert(X, 0, values=1, axis=1)

        return X.dot(self.W)


if __name__ == "__main__":

    # Obtain a simple regression setting with a little noise
    X, y, coef = make_regression(n_samples=100, n_features=1, noise=3,
                                 coef=True, random_state=0)

    # Add some bias to the response
    y += 10

    # Fit a linear regression model with a learning rate of 0.005, including a bias term
    regressor = GDLinearRegressor(0.005, include_bias=True)
    regressor.fit(X, y)

    # Print out the actual and predicted coefficients
    print(10, coef)
    print(regressor.W)

    # Print the MSE
    print(mean_squared_error(y, regressor.predict(X)))




