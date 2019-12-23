import numpy as np
from numpy import linalg as LA
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


class LinearRegression:

    """
        NOTE: This implementation heavily uses normal equations and is therefor not an iterative approach.
    """
    def __init__(self, fit_intercept=True):
        """
        Initializes the model

        :param bool fit_intercept: Whether or not the intercept should be included in the fit
        """
        self.fit_intercept = fit_intercept
        self.B = None

    def fit(self, X, y):
        """
        Computes the betas (weights) on the given data X using y

        :param pandas.DataFrame X: The data on which the betas will be based
        :param pandas.Series y: The response variable that is to be modelled
        """
        # We start with noting that the criterion we want to minimize is the residual sum of squares.
        # The residual sum of squares can be defined as RSS(B) = SUM((yi - f(xi)^2)).
        # In terms of matrices, this results to (y - XB)^T(y - XB), which is a quadratic formula.
        # One can simply deduce that a quadratic formula has just one extreme value; the minima of said function.
        # So to find the coefficients B, such that B minimized the RSS,
        # we simply take the second partial derivative with respect to B and set the first one to zero
        # RSS'_(B)   = -2X.T(y - XB)
        # RSS''_(B)  = 2X.TX
        # Setting it to zero yields:
        # 2X.T(y - XB) = 0 => X.T(y - XB) = 0 => B = (X.TX)^(-1)(X.Ty)

        # Append a column of 1's to X for the betas if set
        if self.fit_intercept:
            X = np.c_[np.ones(len(X)), X]

        # Compute the betas
        self.B = LA.inv((X.T.dot(X))).dot(X.T.dot(y))

    def get_betas(self):
        """
        Obtains the betas to be used to construct the linear function, if required
        """
        return self.B

    def predict(self, X):
        """
        Predicts the response to the given matrix X by doing a dot product on the fit betas

        :param pandas.DataFrame X: The data on which the response will be computed
        """
        if self.B is None:
            raise Exception("The model has not been fit yet!")

        # Append a column of 1's to X for the betas, if set
        if self.fit_intercept:
            X = np.c_[np.ones(len(X)), X]

        # Dot X with the fit betas
        return X.dot(self.B)


if __name__ == "__main__":

    # Generate some synthetic data
    mu, sigma = 0, 2
    X = np.linspace(0, 10, num=100)
    Y = 2*X + np.random.normal(0, 3, 100)

    # Form the input properly
    dataset = pd.DataFrame({"X": X, "Y": Y})
    X = dataset[["X"]]
    y = dataset["Y"]

    # Display the data
    sns.scatterplot(x="X", y="Y", data=dataset)
    plt.show()

    # Define the model and fit it
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    # Obtain the betas
    betas = model.get_betas()

    # Plot the line along with the data
    sns.scatterplot(x="X", y="Y", data=dataset)
    plt.plot([0, 10], [betas[0], betas[1]*10], linewidth=2)
    plt.show()

