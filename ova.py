import numpy as np
import utils


class OneVersusAll(object):
    def __init__(self, num_features, num_class, epsilon=0.01):
        self.num_features = num_features
        self.num_class = num_class
        params = np.random.random((self.num_features, self.num_class))
        self.params = (2 * params - 1) / np.sqrt(self.num_features)
        self.epsilon = epsilon

    def train(self, inputs, labels, episodes, num_batch):
        outputs = utils.to_one_hot(labels, self.num_class)
        for idx in range(episodes):
            batch = np.random.randint(0, len(inputs), num_batch)
            _input = inputs[batch]
            expected = outputs[batch]
            predicted = self.forward(_input)
            error = expected - predicted
            adjustment = np.dot(_input.T, error * utils.sigmoid_output_to_derivative(predicted))
            self.params += self.epsilon * adjustment

    def set_model(self, params):
        self.params = params

    def forward(self, inputs):
        return utils.sigmoid(np.dot(inputs, self.params))

    def predict(self, inputs):
        return np.array(map(utils.from_one_hot, self.forward(inputs)))
