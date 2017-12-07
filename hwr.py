import numpy as np
import utils
from PIL import Image
from hmm import HMM, Inference
from classifier import Predict
from sklearn.cluster import KMeans
from PIL import Image

class HandWritingRecognizer(object):
    def __init__(self, classifier='multinomial_bayes'):
        self.predict = Predict()
        self.classifier = self.predict.__getattribute__(classifier)
        self.HMM = HMM(classifier=classifier)
        self.Inference = Inference(self.HMM)

    def predict_char(self, char_pixels):
        """
        char_pixels = [784, 1] array of character
        """
        return self.HMM.state_labels[self.classifier(char_pixels).squeeze()]

    def predict_segmented_word(self, chars_pixels):
        """
        chars_pixels[n] = [784, 1] array of the n_th character 
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
        nearest = min(utils.get_corpus(), key=lambda w: utils.distance(w, hmm))
        return raw, hmm, nearest

    def cluster_segment(self, image, coords, num_chars):
        """
        given image array, coordinate of non white entries in image and 
        number of characters returns a segmentation based on kmeans cluustering
        """
        kmeans = KMeans(n_clusters=num_chars).fit(coords)
        xcenters = np.sort(kmeans.cluster_centers_[:, 1])
        xedges = xcenters[:-1] + np.diff(xcenters) / 2.
        resize = lambda arr: np.asarray(Image.fromarray(arr).resize([28,28], Image.ANTIALIAS))
        return map(np.ravel, map(resize, np.split(image, xedges, axis=1)))

    def neg_likelihood(self, image, coords, num_chars):
        """
        estimate negative loglikelihood of sequence generated assuming 
        \num_chars\ characters being a true character sequence
        """
        chars_pixels = self.cluster_segment(image, coords, num_chars)
        char_likelihoods = map(self.predict.is_character_probability, chars_pixels)
        return np.mean(char_likelihoods)

    def predict_word(self, img_file='samples/hand.png'):
        """
        raw unsegmented pixels of single word
        """
        img = (255. - np.asarray(Image.open(img_file).convert('L'))) / 255.
        coords = np.array(zip(*np.where(img > 0.5)))
        est_len = min(range(2,15), key=lambda i: self.neg_likelihood(img, coords, i))
        chars_pixels = self.cluster_segment(img, coords, est_len)
        return self.predict_segmented_word(chars_pixels)

    def predict_line(self, pixels):
        pass

