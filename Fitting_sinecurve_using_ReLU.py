import matplotlib.pyplot as plt
import numpy as np


def ReLU(x,weights , bias):
    return np.maximum(np.dot(x, weights) + bias, 0)

def RectifiedLUFitting(x,weights,bias):
    x_for_ReLu_forxaxes = np.arange(-3, 3, 0.01)
    for i, j in zip(weights, bias):
        y = ReLU(x, i, j)  # ReLU applied to the current input
        plt.plot(x_for_ReLu_forxaxes, y)
        x = y




x_for_sine = np.arange(0,10,0.001)
x_for_ReLu= np.arange(-3,3,0.01)
x_for_ReLu_forxaxes = np.arange(-3,3,0.01)
y_for_sine = np.sin(x_for_sine)
weights = np.array([1,0.65])
bias = np.array([0,0.233])

RectifiedLUFitting(x_for_ReLu,weights,bias)
weights = np.array([0.3,0.055])
bias = np.array([0.65,0.96])

RectifiedLUFitting(x_for_ReLu,weights,bias)

#weights = np.array([2.5,1.30])
#bias = np.array([1.045,6.988])

#RectifiedLUFitting(x_for_ReLu,weights,bias)
plt.plot(x_for_sine,y_for_sine)
plt.axhline(y=0, color='r', linestyle='dashed')
plt.savefig("sine.png")
plt.close()