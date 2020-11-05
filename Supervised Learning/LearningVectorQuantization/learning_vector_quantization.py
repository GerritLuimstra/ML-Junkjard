import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from KNearestNeighbours.knn import KNN


class LearningVectorQuantization:

    def __init__(self, n_vectors, max_epochs, learning_rate=0.01, distance_metric=2, debug=False):
        """
        Initializes the Learning Vector Quantization classifier

        :param int n_vectors: The amount of neurons to use (also called 'codebook vectors')
        :param int max_epochs: The amount of epochs to run for
        :param float learning_rate: The learning rate to start with
        :param int distance_metric: The 1/(distance_metric) exponent is used to define the distance metric (based on the Minkowski formula)
        :param bool debug: Whether or not to display debug information
        """

        self.n_vectors = n_vectors
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.debug = debug

        # 1: Manhattan || 2: Euclidean || etc.
        self.p = distance_metric

        # 'Quantized' vectors (also known as codebook vectors)
        self.neurons = None

    def _distance(self, v1, v2):
        """
        Computes the distance between two vectors

        :param np.array v1: The first vector
        :param np.array v2: The second vector
        """
        return sum(abs(v1 - v2)**self.p)**(1/self.p)

    def _get_starting_neurons(self, X):
        """
        Drafts a random samples from the input matrix to obtain 'n_vectors' starting neurons

        :param pandas.DataFrame X: The input features
        """
        return X.sample(self.n_vectors, replace=False)

    def _find_bmu_index(self, vector):
        """
        Attempts to find index of the best matching unit (closest unit) based on the set distance metric

        :param pandas.DataFrame vector: The input vector
        """
        neurons = self.neurons.copy().drop(["__CLASS__"], axis=1)
        distances = [self._distance(vector, neuron) for _, neuron in neurons.iterrows()]
        return np.argmin(distances), min(distances)

    def _schedule_learning_rate(self, epoch):
        """
        Scheduler the learning rate using the following simple formula

        :param int epoch: The current iteration / epoch
        """
        return self.learning_rate * (1 - (epoch/self.max_epochs))

    def _accuracy(self, X, y):
        return (self.predict(X) == y).mean()

    def fit(self, X, y):
        """
        Fits the LVQ to the data

        :param pandas.DataFrame X: The feature vector
        :param pandas.Series y: The response vector
        """

        # See if we have enough vectors
        assert len(X) >= self.n_vectors

        # Create a random set of starting neurons with length = |n_vectors|
        self.neurons = self._get_starting_neurons(X)
        self.neurons["__CLASS__"] = y.loc[self.neurons.index]

        for epoch in range(self.max_epochs):

            # Loop over each vector in the training set
            for index, vector in X.iterrows():

                # Find the closest neuron to the current input vector
                best_matching_unit_index, distance = self._find_bmu_index(vector)
                best_matching_unit = self.neurons.iloc[best_matching_unit_index].copy()
                best_matching_vector = best_matching_unit.copy().drop("__CLASS__")
                current_class = best_matching_unit["__CLASS__"]

                # Calculate the error
                error = (vector - best_matching_vector)

                # Obtain the learning rate
                learning_rate = self._schedule_learning_rate(epoch)

                # If the class matches, move the BMU closer to the data current vector
                if best_matching_unit["__CLASS__"] == y[index]:
                    self.neurons.iloc[best_matching_unit_index] = best_matching_vector + learning_rate * error
                else:
                    # Otherwise, we move the BMU further away from the current vector
                    self.neurons.iloc[best_matching_unit_index] = best_matching_unit - learning_rate * error

                # Remember to fix the class back into the data frame
                self.neurons.iloc[best_matching_unit_index]["__CLASS__"] = current_class

            if self.debug:
                print('>epoch=%d, learning_rate=%.3f, error=%.3f' % (epoch, self._schedule_learning_rate(epoch), self._accuracy(X, y)))

    def predict(self, X):
        """
        Predicts a response for the given feature vector

        :param pandas.DataFrame X: The feature vector
        """
        predictions = []

        # Loop over each vector
        for _, vector in X.iterrows():

            # Find the closest neuron
            closest_neuron = self.neurons.iloc[self._find_bmu_index(vector)[0]]

            # Obtain its class as the prediction
            predictions.append(closest_neuron["__CLASS__"])

        return np.array(predictions)


if __name__ == "__main__":

    # Generate some synthetic data
    # Load in the IRIS dataset
    dataset = datasets.load_iris()
    dataset = pd.DataFrame(data=np.c_[dataset['data'], dataset['target']],
                           columns=dataset['feature_names'] + ['target'])
    dataset = dataset[["petal width (cm)", "petal length (cm)", "target"]]

    # Plot the data
    sns.set()
    sns.scatterplot(x="petal width (cm)", y="petal length (cm)", hue="target", data=dataset)
    plt.show()

    # Prepare the data
    X = dataset[["petal width (cm)", "petal length (cm)"]]
    y = dataset["target"]

    # Split it in a train and test set
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.33, random_state=42)

    # Train the model on the test set
    model = LearningVectorQuantization(n_vectors=20, max_epochs=6, learning_rate=0.1, distance_metric=2, debug=True)
    model.fit(X_train, y_train)

    # Also train a kNN model
    knn_model = KNN(k=5, p=2)
    knn_model.fit(X_train, y_train)

    # Obtain the predictions on the test set
    predictions = model.predict(X_test)
    knn_predictions = knn_model.predict(X_test)

    # Obtain the base line accuracy
    print("BASE LINE ACCURACY:", max(y_test.value_counts())/len(y_test))
    print()

    # Obtain the accuracies
    print("TEST SET ACCURACY LVQ")
    print((predictions == y_test).mean())
    print()
    print("TEST SET ACCURACY KNN")
    print((knn_predictions == y_test).mean())

    # It is cool to see that with *just* 20 vectors, we can approximate k-NN's predictions quite well!



