import numpy as np
from random import choice
# import matplotlib.pyplot as plt
from HWR import HandWritingRecognizer

HWR = HandWritingRecognizer()

char_label_map = dict()
classes = map(str, range(10)) + map(chr, range(97, 123) + range(65, 91))
for i, c in zip(range(62), classes):
    char_label_map[str(c)] = i
    char_label_map[i] = str(c)

def word_to_chars_pixels(word):
    chars = np.load('data/chars.npz')
    test_data, test_labels = chars["test_data"], chars["test_labels"]
    chars_pixels = list()
    for char in word:
        chars_pixels.append(test_data[choice(np.where(test_labels==char_label_map[char])[0])])
    return chars_pixels

def test_word_to_chars_pixels(word='AppLe012'):
    chars_pixels = word_to_chars_pixels(word)
    fig = plt.figure()
    num_chars = len(chars_pixels)
    for i in range(num_chars):
        ax = fig.add_subplot(1, num_chars, i+1)
        ax.imshow(chars_pixels[i].reshape([28,28]), cmap=plt.get_cmap('hot'))
    plt.show()

def test_predict_segmented_word():
    words = ["apple", "crib" ,"dip", "pot", "tree", "stay", "nightly"]
    predictions = list()
    for word in words:
        predictions.append(HWR.predict_segmented_word(word_to_chars_pixels(word)))
    print('{:^10}{:^10}{:^10}'.format('True', 'Raw', 'HMM'))
    print('-' * 30)
    for word, prediction in zip(words, predictions):
        print('{:^10}{:^10}{:^10}'.format( word, prediction[0], prediction[1]))

# test_word_to_chars_pixels()
test_predict_segmented_word()



