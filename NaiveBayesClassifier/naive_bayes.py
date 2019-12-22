import pandas as pd
import numpy as np


class NaiveBayesClassifier:

    def __init__(self):
        self.p_y = None
        self.p_x_y = None
        self.p_x = None
        self.class_names = None


    """
        Fits according to the bayes algorithm
        
        P(y, X) = P(X, y)*P(y)/P(X)
    """

    def fit(self, X, y):

        # Find the class names
        self.class_names = y.unique()

        # Compute the class probabilities

        p_y = []
        for class_name in self.class_names:
           p_y.append(sum(y == class_name))
        p_y = np.array(p_y) / len(y)

        # Compute the probability of X, given a class
        p_x_y = {}

        for feature in X.columns:

            p_x_y[feature] = {}

            for x_value in X[feature].unique():

                p_x_y[feature][x_value] = []

                # Find all the distinct possibilities of y
                for y_value in self.class_names:

                    # Find the indices of when y = y_value
                    y_indices = y[y == y_value].index

                    x_value_counts = X.iloc[y_indices][feature].value_counts()

                    if x_value in x_value_counts.index:
                        p_x_y[feature][x_value].append(x_value_counts[x_value] / len(y_indices))
                    else:
                        p_x_y[feature][x_value].append(0)

        p_x = {}
        for feature in X.columns:
            p_x[feature] = {}

            # Compute the value counts
            value_counts = X[feature].value_counts()

            for x_value in value_counts.index:
                p_x[feature][x_value] = value_counts[x_value] / len(X)

        self.p_x = p_x
        self.p_y = p_y
        self.p_x_y = p_x_y

    def predict(self, X):

        predictions = []

        for _, row in X.iterrows():

            # Obtain the multiplicative P(X, y) for each class
            prob_x_y = [1] * len(self.class_names)
            for feature in X.columns:
                for index, value in enumerate(self.class_names):
                    prob_x_y[index] *= self.p_x_y[feature][row[feature]][index]
            prob_x_y = np.array(prob_x_y)

            # Obtain P(y)
            prob_y = np.array(self.p_y)

            # Obtain P(x)
            prob_x = 1
            for feature in X.columns:
                prob_x *= self.p_x[feature][row[feature]]

            # Obtain the bayes criterion for both classes
            bayes_criterion = (prob_x_y * prob_y) / prob_x

            # Make the prediction
            prediction = self.class_names[np.argmax(bayes_criterion)]

            # Add the prediction
            predictions.append(prediction)

        return np.array(predictions)


if __name__ == "__main__":

    golf_dataset = {
        "outlook": ["sunny", "sunny", "overcast", "rainy", "rainy", "rainy", "overcast", "sunny", "sunny", "rainy", "sunny", "overcast", "overcast", "rainy"],
        "temperature": ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild", "cool", "mild", "mild", "mild", "hot", "mild"],
        "humidity": ["high", "high", "high", "high", "normal", "normal", "normal", "high", "normal", "normal", "normal", "high", "normal", "high"],
        "windy": [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
        "play": ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"]
    }

    X1 = ["bark", "bark", "miauw", "miauw"]
    X2 = ["hard", "hard", "soft", "hard"]
    y = ["dog", "dog", "cat", "cat"]
    cat_dog_dataset = pd.DataFrame({"sound": X1, "paw": X2, "class": y})    #
    cat_dog_X = cat_dog_dataset[["sound", "paw"]]
    cat_dog_y = cat_dog_dataset["class"]

    golf_dataset = pd.DataFrame(golf_dataset)
    golf_X = golf_dataset[["outlook", "temperature", "humidity", "windy"]]
    golf_y = golf_dataset["play"]

    golf_classifier = NaiveBayesClassifier()
    golf_classifier.fit(golf_X, golf_y)
    cat_dog_classifier = NaiveBayesClassifier()
    cat_dog_classifier.fit(cat_dog_X, cat_dog_y)

    # Make the predictions
    golf_predictions = golf_classifier.predict(golf_X)
    cat_dog_predictions = cat_dog_classifier.predict(cat_dog_X)

    # Obtain the accuracies
    print(sum(golf_predictions == golf_y)/len(golf_predictions))  # 0.92!
    print(sum(cat_dog_predictions == cat_dog_y) / len(cat_dog_predictions))  # 1.0!
