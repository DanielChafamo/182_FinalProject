import numpy as np


class NaiveBayes(object):
    def __init__(self, num_classes, num_features, model_dist):
        self.num_classes = num_classes
        self.num_features = num_features
        self.model_dist = model_dist  # fitting distribution, m=multinomial, g=gaussian
        self.alpha = 0.01  # strength of prior
        self.prior = np.zeros(num_classes)  # belief distribution before data
        self.params = np.zeros([num_classes, num_features])  # -log( P( feature_value = 1 | Class = class) )

    def multinomial_fit(self, counts, alpha):
        for label, features in enumerate(counts):
            self.params[label] = -np.log((features + alpha)/sum(features + alpha))

    def gaussian_fit(self, counts):
        pass

    def train(self, data, labels, alpha):
        counts = np.zeros([self.num_classes, self.num_features])
        for feature_values, label in zip(data, labels):
            self.prior[label] += 1
            for idx, feature_value in enumerate(feature_values):
                if feature_value > 0.5:
                    counts[label][idx] += 1

        self.prior/sum(self.prior)  # normalize count to get prior over classes

        if self.model_dist == 'm':
            self.multinomial_fit(counts, alpha)
        else:
            self.gaussian_fit(counts)

    def test(self, data, labels):
        predicted = list()
        for feature_values in data:
            joint = list()
            for _class in range(self.num_classes):
                total = 0
                for j, feature_value in enumerate(feature_values):
                    if feature_value > .5:
                        total += self.params[_class][j]
                joint.append(total)
            predicted.append(min(range(len(joint)), key=lambda a: joint[a]))
        accuracy = float(sum(predicted == labels)) / len(predicted)
        return predicted, accuracy
