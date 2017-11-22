import cPickle, gzip
from naive_bayes import NaiveBayes

with gzip.open('data/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)

num_train = 500
num_test = 500
alpha = .1

train_data, train_labels = train_set[0][:num_train], train_set[1][:num_train]
test_data, test_labels = test_set[0][:num_test], test_set[1][:num_test]

multinomial_classifier = NaiveBayes(10, 28*28, 'm')
multinomial_classifier.train(train_data, train_labels, alpha)

predictions, accuracy = multinomial_classifier.test(test_data, test_labels)

print(zip(predictions, test_labels))
print('alpha: ', alpha)
print('accuracy: ', accuracy)
