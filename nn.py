import numpy as np
import utils


class NeuralNetwork(object):
    def __init__(self, num_input, num_class, layers=None, alpha=0.01, batch_size=32):
        """
        There are [ len(layers) + 1 ] weight sets

        """
        self.layers = layers        # list of hidden layer sizes [or "load"]
        self.num_class = num_class  # output size
        self.num_input = num_input  # input size
        self.alpha = alpha          # learning rate
        self.batch_size = batch_size
        if self.layers == "load":
            nn = np.load('nn.npz')
            self.weights, self.biases = nn[()]['weights'], nn[()]['biases']
        else:
            self.random_weights_biases()

    def random_weights_biases(self):
        if self.layers is None:
            self.layers = [self.num_class] * 3
        self.weights = []
        weight_dims = [self.num_input] + self.layers + [self.num_class]
        for pre_layer, post_layer in zip(weight_dims[:-1], weight_dims[1:]):
            self.weights.append(np.random.rand(post_layer, pre_layer))
            self.bias.append(np.random.rand(post_layer))

    def forward(self, inputs, a_fxn=utils.sigmoid, layer=0):
        """
        inputs: [n, 784]
        returns list of outputs at every layer
        """
        if layer < len(self.weights) - 1:
            output = np.dot(inputs, self.weights[layer]) + self.bias[layer]
            activation = a_fxn(output)
            return list(activation) + self.forward(activation, a_fxn, layer + 1)
        return [np.dot(inputs, self.weights[layer]) + self.bias[layer]]

    def backprop(self, _input, expected):
        """
        inputs: [n, 784]
        activations: [#hidden_Layers + 1, n, #Neurons_in_layer]
        expected: [n]  
        -------------------
        returns gradient of output wrt every parameter
        """

        activations = [_input] + self.forward(_input)

        expected_dist = utils.to_one_hot(expected, self.num_class)
        generated_dist = utils.softmax(activations[-1])

        deltas = [(expected - activation[-1]) * 1]
        for weight_m, activation_m in reversed(zip(self.weights, activations[:-1])):
            deltas.append(weight_m.T.dot(deltas[-1]) * a_fxn(activation_m, True))
        deltas.reverse()

        d_weights, d_biases = np.zeros_like(self.weights), np.zeros_like(self.biases)
        for i in range(len(self.weights)):
            d_weights[i] = 1./self.batch_size * deltas[i].dot(activations[i].T)
            d_biases[i] = np.mean(deltas[i], axis=1).reshape([-1,1])

        return d_weights, d_biases
        
    def update(self, d_weights, d_biases):
        for l in range(len(self.weights)):
            self.weights[l] -= self.alpha * d_weights[l]
            self.biases[l] -= self.alpha * d_biases[l]

    def predict(self, _input, a_fxn=utils.sigmoid):
        generated_dist = utils.softmax(self.forward(_input, a_fxn=a_fxn)[-1])
        return np.argmax(generated_dist, axis=1)

    def train(self, train_set, num_epochs):
        for i in range(num_epochs):
            for j in range(0, len(train_set[0]), self.batch_size):
                X = train_set[0][j:j+batch_size]
                Y = train_set[1][j:j+batch_size]
                self.update(self.backpropagation(X, Y))
            prediction = self.predict(train_set[0])
            print i, np.mean(prediction == np.argmax(teY, axis=1))

    def save_model(self):
        np.save('models/nn', weights=self.weights, biases=self.biases)

nn = NeuralNetwork(10, 5, [20, 40, 10])
print(nn.predict([range(10)]))


********************
????????????????????
********************
????????????????????
********************
????????????????????
********************

"""Functions and classes for training feedforward neural networks."""

import math
import random

import numpy as np
import scipy.optimize as optim


class FeedforwardNN:
    """A simple feedforward neural network class."""

    def __init__(self, sizes, seed=0):
        self.sizes = list(sizes)
        self.w, self.b = _random_layers(self.sizes, seed=seed)

    def __repr__(self):
        sizes = '[{}]'.format(', '.join([str(s) for s in self.sizes]))
        return 'FeedforwardNN({})'.format(sizes)

    @property
    def num_layers(self):
        return len(self.w) + 1

    def predict(self, x):
        """Apply the network to an input.

        Parameters
        ----------
        x : The input.

        Returns
        -------
        The predicted value at x.

        """
        activ = self.forward(x)
        return activ[-1].squeeze()

    def forward(self, x):
        """Collect activations of hidden units.

        Parameters
        ----------
        x : The input.

        Returns
        -------
        A list containing the input and the activations at layer.

        """
        activ = [np.atleast_1d(x)]

        for w, b in zip(self.w, self.b):
            a = sigmoid(w @ activ[-1] + b)
            activ.append(a)

        return activ

    def backprop(self, x, y):
        """Compute the gradient of the loss using backpropagation.

        Parameters
        ----------
        x : Input data.
        y : Target data.

        Returns
        -------
        Partial derivatives with respect to parameters in each layer.

        """
        activ = self.forward(x)
        yhat = activ[-1].squeeze()

        delta = []

        # Special case for last layer.

        z = self.w[-1] @ activ[-2] + self.b[-1]
        delta.append(-(y - yhat) * sigmoid_jac(z))

        # All remaining layers (except the first).

        for k in range(2, self.num_layers):
            z = self.w[-k] @ activ[-(k + 1)] + self.b[-k]
            d = (delta[-1] @ self.w[-(k - 1)]) * sigmoid_jac(z)
            delta.append(d)

        delta.reverse()

        # Compute gradients

        dw = []
        db = []

        for k in range(1, self.num_layers):
            a = activ[k - 1]
            d = delta[k - 1]
            dw.append(_colv(d) * a)
            db.append(d)

        return dw, db

    def update(self, dw, db, rate):
        """Update model parameters using gradients.

        Parameters
        ----------
        dw : Weight gradients.
        db : Bias gradients.
        rate : Learning rate.

        Returns
        -------
        Nothing, just updates the parameters.

        """
        for k in range(self.num_layers - 1):
            self.w[k] -= rate * dw[k]
            self.b[k] -= rate * db[k]

    def loss(self, dataset):
        """Compute the average loss on a dataset.

        Parameters
        ----------
        dataset : List of x, y pairs.

        Returns
        -------
        A single number.

        """
        losses = [0.5 * np.sum((y - self.predict(x))**2) for x, y in dataset]
        return np.mean(losses)


def learn(model, train, rate, batch_size, epochs):
    """Run stochastic gradient descent to learn model weights.

    Parameters
    ----------
    model : Neural network.
    train : Training instances.
    rate : Learning rate to use.
    batch_size : Number of samples in each batch.
    epochs : Number of epochs (passes through the data).

    Returns
    -------
    Updated model.

    """
    train = list(train)
    num_train = len(train)
    num_batches = math.ceil(num_train / batch_size)

    for i in range(epochs):

        random.shuffle(train)

        for j in range(num_batches):
            first = j * batch_size
            last = (j + 1) * batch_size
            batch = train[first:last]

            dw = [0.0] * (model.num_layers - 1)
            db = [0.0] * (model.num_layers - 1)

            for x, y in batch:
                w, b = model.backprop(x, y)

                for k in range(len(dw)):
                    dw[k] += w[k] / len(batch)
                    db[k] += b[k] / len(batch)

            model.update(dw, db, rate)

        if (i + 1) % 10 == 0:
            print(model.loss(train))

    return model


def _random_layers(sizes, scale=0.1, seed=0):
    """Randomly initialize layer weights and biases.

    Parameters
    ----------
    sizes : A list of layer sizes.
    scale : The standard deviation of the initialization distribution.
    seed : Random number generator seed.

    Returns
    -------
    A list of weights corresponding to the layers mapping between
    hidden units.

    """
    rng = np.random.RandomState(seed)

    weights = []
    biases = []
    for n_in, n_out in zip(sizes, sizes[1:]):
        w = rng.normal(scale=scale, size=(n_out, n_in))
        b = rng.normal(scale=scale, size=n_out)
        weights.append(w)
        biases.append(b)

    return weights, biases


def learn_neuron(x, y):
    """Learn the weights of a single neuron.

    Parameters
    ----------
    x : Example inputs.
    y : Example outputs.

    Returns
    -------
    A vector of weights parameterizing a neuron.

    """
    x = np.asarray(x)
    y = np.asarray(y)

    def loss(w):
        yhat = np.array([neuron(w, x) for x in x])
        return 0.5 * np.mean((y - yhat)**2)

    def loss_jac(w):
        yhat = np.array([neuron(w, x) for x in x])
        dw = (yhat - y)[:, None] * np.array([neuron_jac(w, x) for x in x])
        return np.mean(dw, axis=0)

    w = np.random.normal(scale=0.01, size=len(x[0]))
    solution = optim.minimize(loss, w, jac=loss_jac, method='CG')

    return solution['x']


def neuron(w, x):
    """A single neuron activation.

    Parameters
    ----------
    w : The weight vector.
    x : The input vector.

    Returns
    -------
    The neuron activation value.

    """
    return sigmoid(w @ x)


def neuron_jac(w, x):
    """The jacobian of the neuron with respect to w.

    Parameters
    ----------
    w : The weight vector.
    x : The input vector.

    Returns
    -------
    The jacobian of the neuron.

    """
    return sigmoid_jac(w @ x) * x


def sigmoid(x):
    """The sigmoid function.

    Parameters
    ----------
    x : A scalar or ndarray.

    Returns
    -------
    The sigmoid evaluated at x (or its elements if it is an ndarray).

    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_jac(x):
    """The derivative of the sigmoid function.

    Parameters
    ----------
    x : A scalar or ndarray.

    Returns
    -------
    The derivative with respect to x (or each of its elements).

    """
    y = sigmoid(x)
    return y * (1.0 - y)


def add_bias(x):
    """Add a bias (intercept) to the inputs."""
    b = np.ones((len(x), 1))
    return np.hstack((b, x))


def rescale(x, low, high):
    """Rescale values to lie between 0 and 1.

    Parameters
    ----------
    x : Array of values.
    low : The unscaled lower bound.
    high : The unscaled upper bound.

    Returns
    -------
    An array with rescaled values.

    """
    return (x - low) / (high - low)


def _rowv(x):
    """View array as a row vector."""
    return x[None, :]


def _colv(x):
    """View array as a column vector."""
    return _rowv(x).T
