import numpy as np
from classifier import Predict
from segment import Segment
import utils

s = Segment('hello.png')

# print s.segment()
d = s.segment()
# print d
d = np.asarray(d)
# np.savez('data.npz', data=d)
# chars = np.load('data/data.npz')
predict= Predict()
# print p.gaussian_bayes(d)
# print len(d)
for i in range(len(d)):
  d[i] = d[i].reshape([1,-1])
  d[i] = (255.-d[i])/255.
  print utils.char_label_map[predict.multinomial_bayes(d[i])[0]]
