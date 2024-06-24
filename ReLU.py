import matplotlib
import numpy as np


import matplotlib.pyplot as plt

def ReLU(x,weights , bias):
    return np.maximum(np.dot(x, weights) + bias, 0)

x = np.arange(-10,10,1)
y1 = ReLU(x,-1,0.5)
y2 = ReLU(y1,-2,1)
plt.plot(x,y2)
plt.savefig("PLOT1.png")
plt.close()
