# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version

"""
{
    Tools to plot Activation functions in CNN, the figure is to use in Paper.
}
{License_info}
"""

# Futures

# […]

# Built-in/Generic Imports
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# […]

# Libs
# import pandas as pd # Or any other
# […]

# Own modules
# from {path} import {class}
# […]

x = np.arange(-5, 5, 0.01)

def plot(func, yaxis=(-1.4, 1.4)):
    plt.ylim(yaxis)
    plt.locator_params(nbins=5)
    plt.title('TanH')
    plt.text(-3, 1, r'$f(x)=\frac{2}{1+e^{-2x}}-1    $', fontsize=15)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.axhline(lw=1, c='black')
    plt.axvline(lw=1, c='black')
    plt.grid(alpha=0.4, ls='-.')
    plt.box(on=None)
    plt.plot(x, func(x), c='r', lw=3)
    plt.show()

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return 2 / (1+np.exp(-2*x)) - 1

def print_relu():
    relu = np.vectorize(lambda x: x if x > 0 else 0, otypes=[np.float])
    plot(relu, yaxis=(-0.4, 1.4))


def main_func():
    
    plot(tanh)


if __name__ == '__main__':
    main_func()