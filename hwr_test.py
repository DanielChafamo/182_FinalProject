import numpy as np
from random import choice
import matplotlib.pyplot as plt
import utils
from hwr import HandWritingRecognizer


HWR = HandWritingRecognizer()

def word_to_chars_pixels(word):
    chars = np.load('data/chars.npz')
    test_data, test_labels = chars["test_data"], chars["test_labels"]
    chars_pixels = list()
    for char in word:
        if char == ' ':
            chars_pixels.append(np.zeros(28*28))
            continue 
        ID = choice(np.where(test_labels==utils.char_label_map[char])[0])
        chars_pixels.append(test_data[ID])
    return chars_pixels


def test_word_to_chars_pixels(word='AppLe012'):
    chars_pixels = word_to_chars_pixels(word)
    fig = plt.figure()
    num_chars = len(chars_pixels)
    for i in range(num_chars):
        ax = fig.add_subplot(1, num_chars, i+1)
        ax.imshow(chars_pixels[i].reshape([28,28]), cmap=plt.get_cmap('Greys'))
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def accuracies(true, prediction):
    by_word = sum(prediction == true) / float(len(true))
    by_char, num_chars = 0., 0.
    for idx in range(len(true)):
        by_char += sum(np.array(list(true[idx])) == np.array(list(prediction[idx])))
        num_chars += len(true[idx])
    return by_word, by_char / num_chars


def hmm_versus_raw(words=utils.get_corpus(), size=1000):
    true, raws, hmms, finals = np.array([]), np.array([]), np.array([]), np.array([])
    idx = np.random.randint(0, len(words), size)
    for i in idx:
        word = list(words)[i]
        raw, hmm, final = HWR.predict_segmented_word(word_to_chars_pixels(word))
        finals = np.append(finals, final)  
        hmms = np.append(hmms, hmm) 
        raws = np.append(raws, raw)
        true = np.append(true, word)
    print_TRH(true, raws, hmms, finals)
    print_accuracies(accuracies(true, raws), accuracies(true, hmms), accuracies(true, finals))
    

def print_TRH(true, raws, hmms, finals):
    print('{:^20}{:^20}{:^20}{:^20}'.format('True', 'Raw', 'HMM', 'Final'))
    print('-' * 80)
    for word, raw, hmm, final in zip(true, raws, hmms, finals):
        print('{:^20}{:^20}{:^20}{:^20}'.format(word, raw, hmm, final))


def print_accuracies(raw_acc, hmm_acc, final_acc):
    print('{:^10}{:^10}{:^10}{:^10}'.format('', 'Raw[%]', 'HMM[%]', 'Final[%]'))
    print('-' * 30)
    print('{:^10}{:^10.2f}{:^10.2f}{:^10.2f}'
          .format('By word', raw_acc[0]*100, hmm_acc[0]*100, final_acc[0]*100))
    print('{:^10}{:^10.2f}{:^10.2f}{:^10.2f}'
          .format('By char', raw_acc[1]*100, hmm_acc[1]*100, final_acc[1]*100))


# test_word_to_chars_pixels('handwriting')
# hmm_versus_raw(['under', 'the', 'spreading', 'chesnut', 'tree',], 2)
hmm_versus_raw(size=10)

