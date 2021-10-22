# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 18:28:16 2021

@author: @emrenuray
"""

import numpy as np
import matplotlib.pyplot as plt

# Creating noisy sine wave with gaussian.
def myfunc(points, x):
    gauss = np.random.normal(0, 0.05, points)
    myfunc = np.sin(2*np.pi*x) + gauss
    return myfunc

np.random.seed(1)
i=0
while (i<6):
# ? points on the noisy sine wave.
    point_num = [[4,1],[8,2],[16,4],[40,10],[80,20],[800,200]]
    x_values = np.sort(np.random.random((point_num[i][0],)))
    y_values = myfunc(point_num[i][0], x_values)

# Curve fitting
    order = 10
    fit_values = np.polyfit(x_values, y_values, order)
    my_polynom = np.poly1d(fit_values)
    y_fitted = my_polynom(x_values)

# Testing
    x_test = np.sort(np.random.random((point_num[i][1],)))
    y_test = myfunc(point_num[i][1], x_test)

# Plotting
    plt.figure(figsize=(20, 10))
    plt.plot(x_values, y_values, "b-", x_values, y_fitted, "r*", x_test, y_test, "go")
    plt.legend(('Noisy sin wave', 'Training model', "Test model"))
    plt.savefig("Figure"+str(i)+".png")
    i+=1
