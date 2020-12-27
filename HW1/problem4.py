# -*- coding: utf-8 -*-
"""
SCMP Assignment 1 Problem 3

@author: harsh
"""
from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt

x = 1
hspace = np.array([math.pow(10,i) for i in range(-16,1)])

def derv1(x,h,f=np.tan):
	return ((f(x+h)-f(x))/h)

def derv2(x,h,f=np.tan):
	return ((f(x+h)-f(x-h))/(2*h))

def absderv(x):
	return np.power(1/np.cos(x),2)

same = True

yderv1 = np.array([abs(derv1(x,h)-absderv(x)) for h in hspace])
yderv2 = np.array([abs(derv2(x,h)-absderv(x)) for h in hspace])

if not same: plt.subplot(2, 1, 1)
plt.title('Finite Difference Approximation')
plt.plot(np.log(hspace), np.log(yderv1),marker='',label='Finite Difference',linestyle='-')
plt.ylabel('Log Absolute Error')
if not same: plt.subplot(2, 1, 2)
plt.title('Centered Difference Approximation')
plt.plot(np.log(hspace), np.log(yderv2),marker='',label='Centered Difference',linestyle='-')
plt.xlabel('Log h')
plt.ylabel('Log absolute error')
if same: 
    plt.title("f'(x=1) Approximation")
    plt.legend()
plt.savefig('problem4.jpg')
plt.show()
