import numpy as np
import utils
from PIL import Image
from hmm import HMM, Inference
from classifier import Predict
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt

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

    def predict_by_inference(self, chars_pixels):
        predicted_word = list()
        raw = list()
        for char_pixels in chars_pixels:
            predicted_char = self.classifier(char_pixels).squeeze()
            self.Inference.elapse_time()
            self.Inference.observe(predicted_char)
            raw.append(predicted_char)
            predicted_word.append(np.argmax(self.Inference.get_belief_distribution()))
        return raw, predicted_word

    def predict_by_viterbi(self, chars_pixels):
        emissions = map(lambda cp: self.classifier(cp).squeeze(), chars_pixels)
        return emissions, self.Inference.viterbi(emissions)

    def neg_likelihood(self, image, coords, num_chars):
        """
        estimate negative loglikelihood of sequence generated assuming 
        \num_chars\ characters being a true character sequence
        """
        chars_pixels = self.cluster_segment(image, coords, num_chars)
        char_likelihoods = map(self.predict.is_character_probability, chars_pixels)
        return np.mean(char_likelihoods)

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

    def predict_segmented_word(self, chars_pixels, method='predict_by_viterbi'):
        """
        chars_pixels[n] = [784, 1] array of the n_th character 
        returns predictions using classifier only, classifier + hmm and 
        classifier + hmm + dictionary lookup 
        """
        raw, predicted_word = self.__getattribute__(method)(chars_pixels)
        hmm = "".join([self.HMM.state_labels[state] for state in predicted_word])
        raw = "".join(map(lambda r: self.HMM.state_labels[r], raw))
        nearest = min(utils.get_corpus(), key=lambda w: utils.distance(w, hmm))
        return raw, hmm, nearest

    def estimate_length(self, img_file='samples/hand.png'):
        """
        returns number of clusters that result in highest mean probability of 
        segments being characters
        """ 
        img = (255. - np.asarray(Image.open(img_file).convert('L'))) / 255.
        coords = np.array(zip(*np.where(img > 0.5)))
        return min(range(2, 15), key=lambda i: self.neg_likelihood(img, coords, i))

    def display_segmented(self, img_file='samples/hand.png', length=None): 
        """
        displays result of segmenting using kmeans clusturing
        uses estimated length if length is not provided
        """
        if length is None:
            length = self.estimate_length(img_file)
        img = (255. - np.asarray(Image.open(img_file).convert('L'))) / 255.
        coords = np.array(zip(*np.where(img > 0.5)))
        chars_pixels = self.cluster_segment(img, coords, length)
        fig = plt.figure()
        num_chars = len(chars_pixels)
        for i in range(num_chars):
            ax = fig.add_subplot(1, num_chars, i+1)
            ax.imshow(chars_pixels[i].reshape([28,28]), cmap=plt.get_cmap('Greys'))
            ax.axis('off')
        plt.subplots_adjust(wspace=.5, hspace=0)
        plt.show()

    def predict_word(self, img_file='samples/hand.png'):
        """
        raw unsegmented pixels of single word
        """
        img = (255. - np.asarray(Image.open(img_file).convert('L'))) / 255.
        coords = np.array(zip(*np.where(img > 0.5)))
        est_len = min(range(2, 15), key=lambda i: self.neg_likelihood(img, coords, i))
        chars_pixels = self.cluster_segment(img, coords, est_len)
        return self.predict_segmented_word(chars_pixels)


    def predict_line(self, pixels):
        pass

    def neg_prob(self, data):
        neg = self.predict.is_character_probability(data)
        # print(": ", neg)
        return neg

"""
hwr = HandWritingRecognizer()
hwr.predict_word('samples/temp.png')

"""

