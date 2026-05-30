# encoding=utf-8

import numpy as np
import matplotlib.pylab as plt


def tanh(x):
    return (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))

print(round(tanh(3.1415926),2))


X = np.arange(-5.0, 5.0, 0.1)
Y = tanh(X)
plt.plot(X, Y)
plt.ylim(-1.1, 1.1)
plt.show()



