import numpy as np


char_label_map = dict()
classes = map(str, range(10)) + map(chr, range(97, 123) + range(65, 91))
for i, c in zip(range(62), classes):
    char_label_map[str(c)] = i
    char_label_map[i] = str(c)

def softmax(log):
    raw = np.exp(log - np.max(log))
    return raw / np.sum(raw, axis=0)


def to_one_hot(labels, num_classes):
    one_hot = np.zeros([len(labels), num_classes])
    one_hot[range(len(one_hot)), labels] = 1
    return one_hot


def from_one_hot(one_hot):
    return np.argmax(one_hot)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_output_to_derivative(x):
    return x * (1 - x)


def ReLU(x):
    x[ x < 0 ] = 0
    return x


def ReLU_derivative(x):
    x[ x > 0 ] = 1
    return x


def accuracy(predictions, true):
    return np.mean(true == predictions)


def normalize(raw):
    return raw / np.sum(raw, axis=0)


def get_corpus():
    corpus = set()
    with open('data/en_dict.txt') as f:
        for word in f:
            if '\'' not in word:
                corpus.add(word.strip('\r\n'))
    return corpus
    