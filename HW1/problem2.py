# -*- coding: utf-8 -*-
"""
SCMP Assignment 1 Problem 2

@author: harsh
"""

from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt

def sterling(n):
    return math.sqrt((2*math.pi*n))*math.pow((n/math.e),n)
def factorial(n):
    return math.factorial(n)
def abserr(n):
    return abs(factorial(n)-sterling(n))
def relerr(n):
    return abs(factorial(n)-sterling(n))/factorial(n)

x = np.array([x for x in range(1,11)])
abser = np.array([abserr(x) for x in range(1,11)])
reler = np.array([relerr(x) for x in range(1,11)])

plt.subplot(2, 1, 1)
plt.plot(x, abser,marker='o',label='Absolute Error',linestyle='')
plt.title('Errors of sterling approximation')
plt.ylabel('Absolute error')

plt.subplot(2, 1, 2)
plt.plot(x, reler,marker='x',label='Relative Error',linestyle='')
plt.xlabel('x')
plt.ylabel('Relative Error')
plt.savefig('problem2.jpg')
plt.show()
    