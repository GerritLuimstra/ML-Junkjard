import numpy as np


class HardCodedNeuralNetwork:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.input_hidden_weights = np.random.normal(size=(2, 1))

    def relu(self, inputs):
        return max(0, inputs) #return 1/(1 + np.exp(-inputs))

    def relu_grad(self, inputs):
        return np.array(list(map(lambda x: 0.01 if x < 0 else 1, inputs)))

    def forward(self, x_vector):

        # Compute the gradients
        self.hidden_output_delta = x_vector

        # Feed it into the first layer
        hidden_layer = self.input_hidden_weights.T.dot(x_vector)

        # Compute the gradient
        self.relu_gradiant = self.relu_grad(hidden_layer)

        # ReLU
        output = self.relu(hidden_layer)



        return output

    def backward(self, output, actual):

        print("DIFF", output - actual)


        delta_loss_output = -2 * (actual - output)
        delta_relu = delta_loss_output * self.relu_gradiant
        delta_w = delta_relu * self.hidden_output_delta.reshape((2, 1))

        print("DIFF", output - actual)
        print(self.input_hidden_weights)
        print(delta_relu)
        print(delta_w)


        self.input_hidden_weights -= self.learning_rate * delta_w

    def fit(self, x1, x2, y):
        pass

    def predict(self, x1, x2):
        pass


if __name__ == "__main__":

    X = np.array([[178, 22]])
    y = np.array([75])

    nn = HardCodedNeuralNetwork(0.001)

    for i in range(100):

        # Perform a forward pass
        fw = nn.forward(X[0])
        nn.backward(fw, y[0])
