# encoding=utf-8

import numpy as np
import matplotlib.pylab as plt


def relu(x):
    return np.maximum(0, x)

print(round(relu(3.1415926),2))

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1.0, 5.5)
plt.show()



