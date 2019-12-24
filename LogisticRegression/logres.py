import numpy as np
import pandas as pd


class GDLogisticRegression:

    def __init__(self, learning_rate, threshold=0.5, iterations=300, tolerance=0.01, debug=False):
        """
        Initializes the model

        :param float learning_rate: The rate at which the weights are updated
        :param float threshold: The decision boundary threshold (default = 0.5)
        :param int iterations: The amount of iteration gradient descent should run
        :param float tolerance: The residual change of the gradients at which we can stop
        :param bool debug: Whether or not to show intermediate debug information
        """
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.iterations = iterations
        self.tolerance = tolerance
        self.debug = debug
        self.W = None

    def _sigmoid(self, z):
        """
        The sigmoid activation function

        :param pd.Series z : The input to the sigmoid
        """
        return 1/(1 + np.exp(-z))

    def _gradient(self, X, pred, y):
        """
        The gradient of our loss function
        :param pd.DataFrame X : The data to use in the prediction
        :param pd.Series pred : The current predictions by the model on the data X
        :param pd.Series y : The response labels for the data X
        """
        return np.dot(X.T, pred - y)

    def _update_weights(self, X, y, W):
        """
        Computes the gradient of the loss function and updates the weights accordingly

        :param pd.DataFrame X : The data to use in the prediction
        :param pd.Series y : The response labels for the data X
        :param np.array W : The weights that are to be updated
        """

        # Obtain the predictions
        predictions = self.predict_proba(X)

        # Compute the gradient
        gradient = self._gradient(X, predictions, y)

        # Update the weights
        W -= self.learning_rate * gradient

        return W

    def _binary_log_loss(self, X, y):
        """
        Computes the Binary Log Loss based on the current weights and X, y

        :param pd.DataFrame X : The data to use in the prediction
        :param pd.Series y : The response labels for the data X
        """
        h = self.predict_proba(X)
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).sum() / len(X)

    def fit(self, X, y):
        """
        Fits the model to the given data X, using the response y

        :param pd.DataFrame X : The data to fit the model on
        :param pd.Series y : The response labels for the data X
        """

        # Create an empty weight matrix
        self.W = np.random.normal(1, 3, X.shape[1])

        for i in range(self.iterations):

            # Update the weights
            W_prime = self._update_weights(X, y, self.W.copy())

            # Stop if we are not making any substantial progress
            if (abs(self.W - W_prime)).sum() < self.tolerance:
                break

            # Update the weights
            self.W = W_prime

            # Compute the cost
            cost = self._binary_log_loss(X, y)

            if self.debug and (i+1) % 100 == 0:
                print(cost)

    def predict_proba(self, X):
        """
        Predicts the response probabilities based on the input X

        :param pd.DataFrame X : The data to base the responses on
        """
        if self.W is None:
            raise Exception("This model has not been fit yet!")

        return self._sigmoid(X.dot(self.W))

    def predict(self, X):
        """
        Predicts the response classes based on the input X

        :param pd.DataFrame X : The data to base the responses on
        """
        return self.predict_proba(X) > self.threshold


if __name__ == "__main__":

    # Create some synthetic data
    X1 = [4.85, 8.62, 5.43, 9.21]
    X2 = [9.63, 3.23, 8.23, 6.34]
    y = [1, 0, 1, 0]

    # Munge the data in the right form
    dataset = pd.DataFrame({"Studied": X1, "Slept": X2, "Passed": y})
    X = dataset[["Studied", "Slept"]]
    y = dataset["Passed"]

    # Create a classifier
    logres = GDLogisticRegression(learning_rate=0.05, tolerance=0.005, iterations=500, debug=False)
    logres.fit(X, y)

    # Print out the predictions
    print(logres.predict(X))  # Correct!

    # Print the accuracy!
    print(sum(logres.predict(X) == y)/len(X))  # 1.0! :)
