import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random


class Perceptron:

    def __init__(self, learning_rate=0.01):
        """
        Initializes the model

        :param float learning_rate: The rate at which the weight vector is updated (default=0.01)
        """
        self.learning_rate = learning_rate
        self.W = None

    def _has_converged(self, X, y):
        """
        Whether the model has converged yet.
        In the context of the perceptron this means not classifying an instance wrongly

        :param np.array X : The data to fit the model on
        :param pd.Series y : The response labels for the data X
        """
        return np.array_equal(self.predict(X, add_bias=False), np.array(y))

    def fit(self, X, y):
        """
        Fits the model to the given data X, using the response y

        :param pd.DataFrame X : The data to fit the model on
        :param pd.Series y : The response labels for the data X
        """
        # Append a column of 1's to X for the betas
        X = np.c_[np.ones(len(X)), X.copy()]

        # Start with a set of random weights
        self.W = np.random.normal(0, 3, X.shape[1])

        # Loop until we have converged
        while not self._has_converged(X, y):

            # Obtain a random sample from X
            random_index = random.randint(0, X.shape[0] - 1)
            x = X[random_index]

            # Compute the prediction
            prediction = self.predict(np.array([x]), add_bias=False)[0]

            # If our model predicts 0, but it is actually 1,
            # we have to tilt the line away from the point
            if y[random_index] == 1 and prediction == 0:
                self.W = self.W + self.learning_rate * np.array(x)

            # If our model predicts 1, but it is actually 0,
            # we have to tilt the line away from the point
            if y[random_index] == 0 and prediction == 1:
                self.W = self.W - self.learning_rate * np.array(x)

    def get_weight_vector(self):
        """
        If the user so desires, returns the weight vector
        """
        return self.W

    def predict(self, X, add_bias=True):
        """
        Predicts the response classes based on the input X

        :param pd.DataFrame||np.array X : The data to base the responses on
        :param boolean add_bias : Whether or not to add the bias term to the input matrix X
        """

        if self.W is None:
            raise Exception("The model is not fit yet!")

        # Append a column of 1's to X for the betas
        if add_bias:
            X = np.c_[np.ones(len(X)), X]

        return X.dot(self.W.T) >= 0


if __name__ == "__main__":

    # Setup a simple dataset
    X1 = [-2, -1, -1, 1, 2, 3]
    X2 = [2, 2, 1, 3, 2, 3]
    y = [0, 0, 0, 1, 1, 1]
    dataset = pd.DataFrame({"X1": X1, "X2": X2, "class": y})
    X = dataset[["X1", "X2"]]
    y = dataset["class"]

    # Plot the data
    sns.set()
    sns.scatterplot(x="X1", y="X2", style="class", data=dataset)
    plt.show()

    # Fit the model
    perceptron = Perceptron(learning_rate=0.01)
    perceptron.fit(X, y)

    # Verify whether the response vector is equal to the actual labels
    assert np.array_equal(perceptron.predict(X), y)
