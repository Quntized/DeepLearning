import matplotlib.pyplot as plt
import numpy as np

def function(x):
    return x**3
def der_func(x):
    return 3 * x**2
def Iteration():
    x = np.linspace(-2,6,0.01)
    y = function(x)
    learning_rate = 0.01
    current_pos = (10,function(10))
    for _ in range(1000):
        new_x = current_pos[0] - learning_rate * der_func(current_pos[0])
        current_pos = (new_x, def_func(new_x))
        plt.plot(x,y)
        plt.scatter(current_pos[0],current_pos[1],color='red')
        plt.pause(0.01)
        plt.clf()