#!/usr/bin/env python3
"""
Module to create a neural network
"""
import numpy as np


class NeuralNetwork:
    """
    Class that defines a neural network with one hidden layer
    performing binary classification
    """

    def __init__(self, nx, nodes):
        """
        class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return (self.__W1)

    @property
    def b1(self):
        return (self.__b1)

    @property
    def A1(self):
        return (self.__A1)

    @property
    def W2(self):
        return (self.__W2)

    @property
    def b2(self):
        return (self.__b2)

    @property
    def A2(self):
        return (self.__A2)

    def forward_prop(self, X):
        """
        calculates the forward propagation of the neural network
        """

        z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + (np.exp(-z1)))

        z2 = np.matmul(self.W2, self.__A1) + self.b2
        self.__A2 = 1 / (1 + (np.exp(-z2)))

        return (self.A1, self.A2)

    def cost(self, Y, A):
        """
        calculates the cost of the model using logistic regression
        In binary classification, the output layer has a single node
        that represents the probability of the positive class.
        Thus, the cost function is calculated based on the output of this node
        (i.e., A) and the actual labels (i.e., Y).
        """
        m = Y.shape[1]
        m_loss = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = (1 / m) * (-(m_loss))
        return (cost)

    def evaluate(self, X, Y):
        """
        evaluates the neural network's predictions
        """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return (prediction, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        calculates one pass of gradient descent on the neural network
        """

        m = Y.shape[1]

        # the error or loss at the output layer (dz2)
        # by taking the difference between the predicted output A2
        # and the actual labels Y
        dz2 = (A2 - Y)

        # the gradient of the cost with respect to W2 (d__W2)
        # using the error dz2 and the output of the hidden layer A1.
        """
        The derivative of the cost with respect
        to the weights of the output layer
        is the product of the derivative of the cost
        with respect to the activations of the output layer (dz2)
        and the derivative of the activations
        of the output layer
        with respect to the weights of the output layer (A1.transpose()).
        """
        d__W2 = (1 / m) * (np.matmul(dz2, A1.transpose()))
        # similarly, by taking the average of the errors over all examples.
        d__b2 = (1 / m) * (np.sum(dz2, axis=1, keepdims=True))

        # the error or loss at the hidden layer (dz1) by taking
        # the dot product of W2 and dz2
        # and element-wise multiplication of the result with A1 and (1 - A1).
        dz1 = (np.matmul(self.W2.transpose(), dz2)) * (A1 * (1 - A1))
        #  the gradient of the cost with respect to W1 (d__W1)
        # using the error dz1 and the input data X.
        d__W1 = (1 / m) * (np.matmul(dz1, X.transpose()))
        d__b1 = (1 / m) * (np.sum(dz1, axis=1, keepdims=True))

        self.__W2 = self.W2 - (alpha * d__W2)
        self.__b2 = self.b2 - (alpha * d__b2)

        self.__W1 = self.W1 - (alpha * d__W1)
        self.__b1 = self.b1 - (alpha * d__b1)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        trains the neuron and updates __W1, __b1, __A1, __W2, __b2, and __A2
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for itr in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return (self.evaluate(X, Y))
