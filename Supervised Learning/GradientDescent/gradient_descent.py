import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


def fun(x, y):
    """
    Computes the function f(x) = x^2 + xy + y^2 at position (x, y)
    :param x: The value of x to evaluate f() at
    :param y: The value of y to evaluate f() at
    :return: the value f(x, y)
    """
    return x**2 + x*y + y**2


def gradient(x, y):
    """
    Computes the gradient vector of f(x) = x^2 + xy + y^2 at position (x, y)

    :param x: The x to evaluate the gradient at
    :param y: The y to evaluate the gradient at
    :return: The gradient vector at position (x, y)
    """
    deriv_x = 2*x + y
    deriv_y = x + 2*y
    return np.array([deriv_x, deriv_y])


class GradientDescent:
    """
        Performs gradient descent on convex functions
    """

    def __init__(self, initial_point, function, gradient, learning_rate, ax):
        """

        :param initial_point: The initial point
        :param function : The function that we want to minimize
        :param gradient: The gradient of the given function
        :param learning_rate: The learning rate of gradient descent
        :param ax: The axis at which to plot the intermediate points at
        """
        self.initial_point = initial_point
        self.function = function
        self.gradient = gradient
        self.current_point = np.array([*initial_point])
        self.learning_rate = learning_rate
        self.epsilon = 10**(-10)
        self.ax = ax

    def descent(self):
        """
        Performs gradient descent
        """

        while True:

            # Compute the next point by using the gradient and the learning rate
            new_point = self.current_point - self.learning_rate * self.gradient(self.current_point[0],
                                                                                self.current_point[1])

            # Stop if there is no substantial progress
            # Note that since in machine learning the generalization error is usually
            # more sought after than the true minimum, this is not a problem
            if np.all(np.isclose(new_point, self.current_point, atol=self.epsilon)):
                break

            # Update the point
            self.current_point = new_point

            # Graph the progress
            x, y = self.current_point[0], self.current_point[1]
            ax.plot([x], [y], [self.function(x, y)], marker='x', color='red')


if __name__ == "__main__":

    # Plot a simple surface for the function f(x) = x^2 + xy + y^2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-10, 10, 0.1)
    X, Y = np.meshgrid(x, y)
    zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z, alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')

    # Generate a random starting point
    initial_point = np.random.rand(1, 2)[0] * 10 * random.choice([1, -1])

    # Do the gradient descent step
    gradient_descent = GradientDescent(initial_point, fun, gradient, 0.1, ax)
    gradient_descent.descent()

    # Plot the final result
    plt.show()

    # Print the final output. This should be roughly (0, 0) since the chosen function is convex.
    print(gradient_descent.current_point)
