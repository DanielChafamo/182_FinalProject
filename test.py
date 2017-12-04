import numpy as np
from classifier import Predict
from segment import Segment

s = Segment('hello.png')

# print s.segment()
d = s.segment()
# print d
d = np.asarray(d)
# np.savez('data.npz', data=d)
# chars = np.load('data/data.npz')
p = Predict()
# print p.gaussian_bayes(d)
# print len(d)
for i in range(len(d)):
  data = d[i].flatten()
  print p.gaussian_bayes(data)
