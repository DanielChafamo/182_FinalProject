import numpy as np
import utils
from hmm import HMM, Inference
from classifier import Predict


class HandWritingRecognizer(object):
    def __init__(self, classifier='multinomial_bayes'):
        predict = Predict()
        self.classifier = predict.__getattribute__(classifier)
        self.HMM = HMM(classifier=classifier)
        self.Inference = Inference(self.HMM)

    def predict_char(self, pixels):
        pass

    def predict_segmented_word(self, chars_pixels):
        """
        pixels[n] = [784, 1] array of the n_th word 
        """
        predicted_word = list()
        raw = list()
        for char_pixels in chars_pixels:
            predicted_char = self.classifier(char_pixels).squeeze()
            self.Inference.elapse_time()
            self.Inference.observe(predicted_char)
            raw.append(self.HMM.state_labels[predicted_char])
            predicted_word.append(np.argmax(self.Inference.get_belief_distribution()))
        hmm = "".join([self.HMM.state_labels[state] for state in predicted_word])
        raw = "".join(raw)
        nearest = self.get_nearest_neighbour(hmm)
        return raw, hmm, nearest

    def predict_word(self, pixels):
        pass

    def get_nearest_neighbour(self, prediction, dictionary=utils.get_corpus()):
        return min(dictionary, key=lambda word: utils.get_distance(word, prediction))

    def predict_line(self, pixels):
        pass



