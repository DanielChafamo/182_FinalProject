import cPickle
from naive_bayes import NaiveBayes

with open('data/mnist.pkl', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)

multinomial_classifier = NaiveBayes(10, 28*28, 'm')
multinomial_classifier.train(train_set[0][:1000], train_set[1][:1000], 0.01)
print(multinomial_classifier.test(test_set[0][:1000], test_set[1][:1000]))
