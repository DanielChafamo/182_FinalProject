import numpy as np
import utils
from classifier import Predict

corpus = utils.get_corpus()

class HMM(object):
    """ classifier provides noisy signal """
    def __init__(self, state_labels=utils.classes, corpus=corpus, classifier='multinomial_bayes'):
        self.state_labels = state_labels
        self.num_states = len(self.state_labels)
        self.states = np.array(range(self.num_states))
        self.prior = self.get_prior()
        self.transition_model = self.get_TM(corpus)
        self.emission_model = self.get_EM(classifier)

    def construct_prior(self, unif=False, alpha=1e-2):
        """ Prior:
                prior[state] = P(S_{0} = state)
            either uniform or gleaned from corpus
        """
        if unif:
            return utils.normalize(np.ones_like(self.states))
        prior = np.zeros_like(self.states) + alpha
        for word in corpus:
            for char in word:
                prior[utils.char_label_map[char]] += 1
        prior = utils.normalize(prior)
        np.save('models/prior.npy', prior)
        return prior

    def construct_TM(self, corpus, alpha=1e-2):
        """ 
        Tranistion Model:
            P(successor | current) indexed as: transitionModel[current][successor]
            P(S_{t} = current | S_{t-1} = successor) 
        Gleaned from frequency of sequences in corpus
        """
        transitionModel = np.zeros([self.num_states, self.num_states]) + alpha
        for word in corpus:
            for char_1, char_2 in zip(word[:-1], word[1:]):
                transitionModel[utils.char_label_map[char_2], utils.char_label_map[char_1]] += 1
        transitionModel = utils.normalize(transitionModel)
        np.save('models/transition_model.npy', transitionModel)
        return transitionModel

    def construct_EM(self, alpha=1e-2):
        """ 
        Emission Model:
            P( evidence | state) indexed as: emissionModel[evidence][state]
            P(E_{t} = evid | S_{t} = state)
        Gleaned from accuracy/crossover rates in test data
        """
        chars = np.load('data/chars.npz')
        test_data, test_labels = chars['test_data'], chars['test_labels']
        emitters = ['multinomial_bayes', 'one_versus_all']
        predict = Predict()
        emissionModels = dict()
        for emitter in emitters:
            EM = np.zeros([self.num_states, self.num_states]) + alpha
            predictions = predict.__getattribute__(emitter)(test_data)
            for state, emission in zip(test_labels, predictions):
                EM[emission, state] += 1
            emissionModels[emitter] = utils.normalize(EM)
        np.save('models/emission_model', emissionModels)
        return emissionModels

    def get_prior(self):
        try:
            return np.load('models/prior.npy')
        except IOError:
            return self.construct_prior()

    def get_TM(self, corpus):
        try:
            return np.load('models/transition_model.npy')
        except IOError:
            return self.construct_TM(corpus)

    def get_EM(self, classifier):
        try:
            return np.load('models/emission_model.npy')[()][classifier]
        except IOError:
            return self.construct_EM()[classifier]


class Inference(object):
    """Exact Inference for specified HMM"""
    def __init__(self, HMM):
        self.HMM = HMM
        self.beliefs = self.HMM.prior

    def elapse_time(self):
        """ 
        obtain one step prediction 
            assumes current beliefs are P(previous|previous_evidence)
            updates beliefs to P(current|previous_evidence) 
        """
        all_possible = np.zeros(self.HMM.num_states)
        for current in self.HMM.states:
            all_possible[current] = sum([self.beliefs[previous] * self.HMM.transition_model[current, previous] for previous in self.HMM.states])
        self.beliefs = utils.normalize(all_possible)

    def observe(self, emission):
        """ 
        reweigh prediction using observed evidence 
            assumes current beliefs are P(current|previous_evidence)
            updates beliefs to P(current|all_evidence)
        """
        emission_model = self.HMM.emission_model[emission]
        all_possible = np.zeros(self.HMM.num_states)
        for state in self.HMM.states:
            all_possible[state] = emission_model[state] * self.beliefs[state]
        self.beliefs = utils.normalize(all_possible)

    def get_belief_distribution(self):
        return self.beliefs

