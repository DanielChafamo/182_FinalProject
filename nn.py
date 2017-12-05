import numpy as np
import utils


class NeuralNetwork(object):
    def __init__(self, num_input, num_class, layers=None, alpha=1e-3, batch_size=32):
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
        self.biases = []
        weight_dims = [self.num_input] + self.layers + [self.num_class]
        for pre_layer, post_layer in zip(weight_dims[:-1], weight_dims[1:]):
            self.weights.append(np.random.rand(pre_layer, post_layer))
            self.biases.append(np.random.rand(post_layer))

    def forward(self, inputs, a_fxn=utils.sigmoid, layer=0):
        """
        inputs: [n, 784]
        returns list of outputs at every layer
        """
        inputs = np.asarray(inputs)
        activations = [inputs]
        for weight, bias in zip(self.weights, self.biases):
            print(activations[-1].shape, weight.shape)
            activations.append(a_fxn(activations[-1].dot(weight) + bias))
        return activations

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

        deltas = [(expected_dist - generated_dist)]
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

    def train(self, train_set, num_epochs, save=False):
        for i in range(num_epochs):
            for j in range(0, len(train_set[0]), self.batch_size):
                X = train_set[0][j:j+self.batch_size]
                Y = train_set[1][j:j+self.batch_size]
                self.update(self.backprop(X, Y))
            prediction = self.predict(train_set[0])
            print i, np.mean(prediction == np.argmax(train_set, axis=1))
        if save:
            np.savez('models/nn', weights=self.weights, biases=self.biases)

nn = NeuralNetwork(28*28, 62, [200, 400, 800, 400, 200])
nn.train(utils.get_train_set(), 100)

# print(nn.predict( np.array([range(20,30), range(1,11)] )))

