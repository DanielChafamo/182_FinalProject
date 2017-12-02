import cPickle
import gzip
import numpy as np

from naive_bayes import MultinomialNaiveBayes, GaussianNaiveBayes
from ova import OneVersusAll


class Train(object):
    def __init__(self, train_set=None, num_features=28*28, num_classes=62):
        self.num_features = num_features
        self.num_classes = num_classes
        if train_set is None:
            chars = np.load('data/chars.npz')
            train_set = chars['train_data'], np.vectorize(int)(chars['train_labels'])
        self.train_data, self.train_labels = train_set

    def multinomial_bayes(self, alpha=0.01, save=False):
        mbayes = MultinomialNaiveBayes(self.num_features, self.num_classes)
        mbayes.train(self.train_data, self.train_labels, alpha)
        if save:
            np.save('models/multinomial_bayes.npy', mbayes.params)
        return mbayes.params

    def gaussian_bayes(self, save=False):
        gbayes = GaussianNaiveBayes(self.num_features, self.num_classes)
        gbayes.train(self.train_data, self.train_labels)
        if save:
            np.save('models/gaussian_bayes.npy', np.array([gbayes.means, gbayes.stdvs]))
        return gbayes.means, gbayes.stdvs

    def one_versus_all(self, episodes, epsilon=0.01, save=False):
        ova = OneVersusAll(self.num_features, self.num_classes, epsilon)
        ova.train(self.train_data, self.train_labels, episodes)
        if save:
            np.save('models/one_versus_all.npy', ova.params)
        return ova.params


class Predict(object):
    def __init__(self, num_features=28*28, num_classes=62):
        self.num_features = num_features
        self.num_classes = num_classes

    def multinomial_bayes(self, data=None, model=None):
        if model is None:
            model = np.load('models/multinomial_bayes.npy')
        mbayes = MultinomialNaiveBayes(self.num_features, self.num_classes)
        mbayes.set_model(model)
        prediction = np.array(map(mbayes.predict, np.atleast_2d(data)))
        return prediction

    def gaussian_bayes(self, data=None, model=None):
        if model is None:
            model = np.load('models/gaussian_bayes.npy')
        gbayes = GaussianNaiveBayes(self.num_features, self.num_classes)
        gbayes.set_model(model[0], model[1])
        prediction = np.array(map(gbayes.predict, np.atleast_2d(data)))
        return prediction

    def one_versus_all(self, data=None, model=None):
        if model is None:
            model = np.load('models/one_versus_all.npy')
        ova = OneVersusAll(self.num_features, self.num_classes)
        ova.set_model(model)
        prediction = ova.predict(np.atleast_2d(data))
        return prediction


"""

train = Train()
train.multinomial_bayes(save=True)
train.gaussian_bayes(save=True)
train.one_versus_all(save=True)





train_set = np.load('data/train_chars.npy')
train_set = np.array(train_set[0].tolist()) , train_set[1]
train = Train(train_set=train_set, num_classes=62)
mbayes_params = train.multinomial_bayes()

test_set = np.load('data/test_chars.npy')
test_set = np.array(test_set[0].tolist()) , test_set[1]
predict = Predict(data=test_set[0], num_classes=62)
prediction = np.array(predict.multinomial_bayes(model=mbayes_params))

sum(prediction)

"""

