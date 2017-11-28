import numpy as np
import matplotlib.pyplot as plt


class Visualize(object):
    def __init__(self, num_classes=10):
        self.num_classes = num_classes

    def kernels(self, params, cmap='hot'):
        fig = plt.figure()
        for i in range(1, self.num_classes + 1):
            ax = fig.add_subplot(2, 5, i)
            plt.axis('off')
            ax.imshow(params[i - 1].reshape([28, 28]), cmap=plt.get_cmap(cmap))
        plt.tight_layout()
        plt.show()

    def accuracy_analysis(self, predicted, true):
        tally = np.zeros([self.num_classes, self.num_classes])
        for idx in range(len(true)):
            tally[true[idx]][predicted[idx]] += 1
        accuracies = [tally[_class][_class] / sum(tally[_class]) for _class in range(self.num_classes)]
        return accuracies, tally


vis = Visualize()

"""
#  ______________ KERNELS ______________
gbayes_means = np.load('models/gBayes/means.npy')
mbayes_params = np.load('models/mBayes/params.npy')
vis.kernels(gbayes_means)
vis.kernels(mbayes_params)


# _______________ ACCURACY ______________
model_file = 'gbayes_prediction.npy'
prediction = np.load('results/predictions/' + model_file)
true = np.load('results/predictions/true.npy')

acc, tally = vis.accuracy_analysis(prediction, true)

plt.imshow(tally, cmap=plt.get_cmap('Greys'))
plt.axis('off')
plt.show()

"""