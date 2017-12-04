import numpy as np
from random import choice
# import matplotlib.pyplot as plt
import utils
from HWR import HandWritingRecognizer


HWR = HandWritingRecognizer()


def word_to_chars_pixels(word):
    chars = np.load('data/chars.npz')
    test_data, test_labels = chars["test_data"], chars["test_labels"]
    chars_pixels = list()
    for char in word:
        chars_pixels.append(test_data[choice(np.where(test_labels==utils.char_label_map[char])[0])])
    return chars_pixels


def test_word_to_chars_pixels(word='AppLe012'):
    chars_pixels = word_to_chars_pixels(word)
    fig = plt.figure()
    num_chars = len(chars_pixels)
    for i in range(num_chars):
        ax = fig.add_subplot(1, num_chars, i+1)
        ax.imshow(chars_pixels[i].reshape([28,28]), cmap=plt.get_cmap('hot'))
    plt.show()


def hmm_versus_raw(words=utils.get_corpus(), upto=1000):
    true, raws, hmms = np.array([]), np.array([]), np.array([])
    start = np.random.randint(len(words) - upto)
    for word in list(words)[start : start + upto]:
        raw, hmm = HWR.predict_segmented_word(word_to_chars_pixels(word))
        hmms, raws, true = np.append(hmms, hmm), np.append(raws, raw), np.append(true, word)
    print_TRH(true, raws, hmms)
    print_accuracies(accuracies(true, raws), accuracies(true, hmms))
    

def print_TRH(true, raws, hmms):
    print('{:^20}{:^20}{:^20}'.format('True', 'Raw', 'HMM'))
    print('-' * 60)
    for word, raw, hmm in zip(true, raws, hmms):
        print('{:^20}{:^20}{:^20}'.format(word, raw, hmm))


def print_accuracies(raw_acc, hmm_acc):
    print('{:^10}{:^10}{:^10}'.format('', 'Raw[%]', 'HMM[%]'))
    print('-' * 30)
    print('{:^10}{:^10.2f}{:^10.2f}'.format('By word', raw_acc[0]*100, hmm_acc[0]*100))
    print('{:^10}{:^10.2f}{:^10.2f}'.format('By char', raw_acc[1]*100, hmm_acc[1]*100))


def accuracies(true, prediction):
    by_word = sum(prediction == true) / float(len(true))
    by_char, num_chars = 0., 0.
    for idx in range(len(true)):
        by_char += sum(np.array(list(true[idx])) == np.array(list(prediction[idx])))
        num_chars += len(true[idx])
    return by_word, by_char / num_chars


# test_word_to_chars_pixels()
# hmm_versus_raw(["apple", "crib" ,"dip", "pot", "tree", "stay", "nightly"], 7)
hmm_versus_raw(upto=100)



