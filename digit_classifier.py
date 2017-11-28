import cPickle
import gzip
import numpy as np

from naive_bayes import MultinomialNaiveBayes, GaussianNaiveBayes
from ova import OneVersusAll

with gzip.open('data/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)


class Train(object):
    def __init__(self, num_train, num_features=28*28, num_classes=10):
        self.num_train = num_train
        self.num_features = num_features
        self.num_classes = num_classes
        self.train_data = train_set[0][:self.num_train]
        self.train_labels = train_set[1][:self.num_train]

    def multinomial_bayes(self, alpha=0.01, save=False):
        mbayes = MultinomialNaiveBayes(self.num_features, self.num_classes)
        mbayes.train(self.train_data, self.train_labels, alpha)
        if save:
            np.save('models/mbayes/params.npy', mbayes.params)
        return mbayes.params

    def gaussian_bayes(self, save=False):
        gbayes = GaussianNaiveBayes(self.num_features, self.num_classes)
        gbayes.train(self.train_data, self.train_labels)
        if save:
            np.save('models/gbayes/means.npy', gbayes.means)
            np.save('models/gbayes/stdvs.npy', gbayes.stdvs)
        return gbayes.means, gbayes.stdvs

    def one_versus_all(self, episodes, epsilon=0.01, save=False):
        ova = OneVersusAll(self.num_features, self.num_classes, epsilon)
        ova.train(self.train_data, self.train_labels, episodes)
        if save:
            np.save('models/ova/params.npy', ova.params)
        return ova.params


class Predict(object):
    def __init__(self, num_test=10000, num_features=28*28, num_classes=10):
        self.num_test = num_test
        self.num_features = num_features
        self.num_classes = num_classes
        self.test_data = test_set[0][:self.num_test]
        self.test_labels = test_set[1][:self.num_test]

    def multinomial_bayes(self, data=None, model=None, save=False, alpha=0.01):
        if data is None:
            data = self.test_data
        if model is None:
            model = np.load('models/mbayes/params.npy')
        mbayes = MultinomialNaiveBayes(self.num_features, self.num_classes)
        mbayes.set_model(model)
        prediction = map(mbayes.predict, data)
        if save:
            np.save('results/predictions/mbayes_prediction.npy', prediction)
        return prediction

    def gaussian_bayes(self, data=None, model=None, save=False):
        if data is None:
            data = self.test_data
        if model is None:
            model = np.load('models/gbayes/means.npy'), np.load('models/gbayes/stdvs.npy')
        gbayes = GaussianNaiveBayes(self.num_features, self.num_classes)
        gbayes.set_model(model[0], model[1])
        prediction = map(gbayes.predict, data)
        if save:
            np.save('results/predictions/gbayes_prediction.npy', prediction)
        return prediction

    def one_versus_all(self, data=None, model=None, save=False):
        if data is None:
            data = self.test_data
        if model is None:
            model = np.load('models/ova/params.npy')
        ova = OneVersusAll(self.num_features, self.num_classes)
        ova.set_model(model)
        prediction = ova.predict(data)
        if save:
            np.save('results/predictions/ova_prediction.npy', prediction)
        return prediction


"""
train = Train(5000)
mbayes_params = train.multinomial_bayes()
gbayes_params = train.gaussian_bayes()
ova_params = train.one_versus_all(200)

predict = Predict()
predict.multinomial_bayes(model=mbayes_params)
predict.gaussian_bayes(model=gbayes)
predict.one_versus_all(model=ova_params)

"""

