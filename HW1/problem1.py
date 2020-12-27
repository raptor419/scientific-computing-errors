# -*- coding: utf-8 -*-
"""
SCMP Assignment 1 Problem 1

@author: harsh
"""
from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt

def p(x):
    return (x-2)**9

def p_exp(x):
    exp = 	(1*(x**9))  		\
    		-(18*(x**8))  		\
    		+(144*(x**7))  		\
    		-(672*(x**6))  		\
    		+(2016*(x**5))  	\
    		-(4032*(x**4))  	\
    		+(5376*(x**3)) 		\
    		-(4608*(x**2)) 		\
    		+(2304*(x**1))  	\
    		-(512*(x**0))
    return exp

xspace = np.array([i/1000 for i in range(1920,2081)])
y_p = np.array([p(x) for x in xspace])
y_pexp = np.array([p_exp(x) for x in xspace])

same = True

if not same: plt.subplot(2, 1, 1)
plt.title('Polynomial Evaluation')
plt.plot(xspace, y_p,marker='+',label='p(x) unexpanded',linestyle='')
plt.ylabel('p(x) unexpanded')
if not same: plt.subplot(2, 1, 2)
plt.plot(xspace, y_pexp,marker='x',label='p(x) expanded',linestyle='')
plt.xlabel('x')
plt.ylabel('p(x) expanded')

if same:
    plt.ylabel('p(x)')
    plt.legend()
plt.savefig('problem1.jpg')
plt.show()
    