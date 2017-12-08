import numpy as np
from scipy.stats import norm


class NaiveBayes(object):
    def __init__(self, num_features, num_classes):
        self.num_features = num_features
        self.num_classes = num_classes
        self.optimal_alpha = 0.01  # strength of prior
        self.prior = np.ones(num_classes)/num_classes  # belief distribution before data

    def fit(self, counts, alpha):
        raise NotImplementedError

    def conditional_p(self, _class, feature, feature_value):
        raise NotImplementedError

    def train(self, data, labels, alpha=.01):
        """
        parse by label and feed to fit
        """
        class_data = dict()
        for _class in range(self.num_classes):
            class_data[_class] = data[labels == _class]
            self.prior[_class] = float(len(class_data[_class])) / len(data)
        self.fit(class_data, alpha)

    def predict(self, feature_values, score=False):
        """
        returns class with minimum log likelihood given feature values
        """
        joint = -np.log(self.prior)
        for _class in range(self.num_classes):
            for feature, feature_value in enumerate(feature_values):
                joint[_class] += self.conditional_p(_class, feature, feature_value)
        minimized = min(range(len(joint)), key=lambda a: joint[a])
        if score: return joint[minimized]
        return minimized

    def tune_alpha(self, train_set, validation_set):
        optimal_alpha = 1e-1
        max_accuracy = 0.
        for alpha in range(0, 1, 0.001):
            self.train(*train_set, alpha)
            prediction = np.array(map(self.predict, validation_set[0]))
            accuracy = np.mean(validation_set[1] == prediction)
            if accuracy > max_accuracy:
                optimal_alpha = alpha
                max_accuracy = accuracy
        return optimal_alpha


class MultinomialNaiveBayes(NaiveBayes):
    def __init__(self, num_features, num_classes):
        super(MultinomialNaiveBayes, self).__init__(num_features, num_classes)
        self.params = np.zeros([num_classes, num_features])

    def set_model(self, params):
        self.params = params

    def conditional_p(self, _class, feature, feature_value):
        if feature_value > .5: return self.params[_class][feature]
        return 0.

    def fit(self, class_data, alpha):
        """
        generates multinomial distribution parameters for P(feature_value | class) 
        """
        for _class, data in class_data.iteritems():
            data[data < 0.5] = 0.
            data[data >= 0.5] = 1.
            counts = np.array(map(sum, data.T)) + alpha
            self.params[_class] = -np.log(counts / sum(counts))


class GaussianNaiveBayes(NaiveBayes):
    def __init__(self, num_features, num_classes):
        super(GaussianNaiveBayes, self).__init__(num_features, num_classes)
        self.means = np.zeros([self.num_classes, self.num_features])
        self.stdvs = np.zeros([self.num_classes, self.num_features])

    def set_model(self, means, stdvs):
        self.means = means
        self.stdvs = stdvs

    def conditional_p(self, _class, feature, feature_value):
        return -norm.logpdf(feature_value, self.means[_class, feature], self.stdvs[_class, feature])

    def fit(self, class_data, alpha):
        """
        generates multivariate normal distribution parameters for P(feature_value | class) 
        """
        for _class, _data in class_data.iteritems():
            self.means[_class] = np.mean(_data, axis=0)
            self.stdvs[_class] = np.var(_data, axis=0) + alpha
