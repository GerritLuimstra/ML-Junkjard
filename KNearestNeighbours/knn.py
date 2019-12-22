import numpy as np
import pandas as pd


class KNN:

    def __init__(self, k, p=2):
        """
        Initializes the nearest neighbour classifier

        :param int k : The amount of neighbours to look at
        :param int p : The 1/p exponent is used to define the distance metric (based on the Minkowski formula)
        """
        self.k = k
        self.p = p
        self.X = None
        self.y = None

    def _distance(self, v1, v2):
        """
        Computes the distance between two vectors

        :param np.array v1: The first vector
        :param np.array v2: The second vector
        """
        return sum(abs(v1 - v2)**self.p)**(1/self.p)

    def fit(self, X, y):
        """
        'Fits' the KNN to the data

        :param pandas.DataFrame X: The feature vector
        :param pandas.Series y: The response vector
        """
        self.X = X
        self.y = y

    def predict(self, X):
        """
        Predicts a response for the given feature vector

        :param pandas.DataFrame X: The feature vector
        """

        if self.X is None:
            raise Exception("The model has not been fit yet!")

        if len(X) < self.k:
            raise Exception("Not enough observation to do the prediction.")

        responses = []

        for _, observation in X.iterrows():

            # Compute the distance to each point in self.X
            distances = [self._distance(np.array(observation), np.array(stored_observation)) for _, stored_observation in self.X.iterrows()]

            # Grab the first k responses of the stored observations that are closest to the given observation
            closest_observation_responses = self.y.iloc[np.argsort(distances)[:self.k]]

            # Find the most dominant response
            most_dominant_response = np.bincount(closest_observation_responses).argmax()

            # Add the response to the responses so far
            responses.append(most_dominant_response)

        return np.array(responses)


if __name__ == "__main__":

    X1 = [0, 1, 1, 2, 3, 3]
    X2 = [1, 1, 0, 3, 3, 2]
    y = [1, 1, 1, 2, 2, 2]

    dataset = pd.DataFrame({"X1": X1, "X2": X2, "class": y})
    X = dataset[["X1", "X2"]]
    y = dataset["class"]

    knn = KNN(k=3, p=2)
    knn.fit(X, y)

    print(knn.predict(X))  # correct! :)


