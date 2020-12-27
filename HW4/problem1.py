"""
@author: harsh
"""
from __future__ import division
import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla

A1 = np.array([[2,3,2],[10,3,4],[3,6,1]])
A2 = np.array([[6,2,1],[2,3,1],[1,1,1]])
X = np.array([[0],[0],[1]])

A = A1

def power_iteration(A, eiv):
    eps = 10**-9
    X = eiv / npla.norm(eiv)
    evali,evalf = 1,0
    
    while(abs(evali-evalf)>eps):
        evali = evalf
        Y = np.dot(A,X)
        X = Y / npla.norm(Y)
        evalf = np.dot(X.T,np.dot(A,X))/np.dot(X.T,X)
    
    return evalf,X

def inverse_power_iteration(A, eiv):
    eps = 10**-9
    X = eiv / npla.norm(eiv)
    evali,evalf = 0,0
    evalf = np.dot(X.T,np.dot(A,X))/np.dot(X.T,X)
    
    while(abs(evali-evalf)>eps):
        evali = evalf
        Y = npla.solve(A,X)
        X = Y / npla.norm(Y)
        evalf = np.dot(X.T,np.dot(A,X))/np.dot(X.T,X)
    
    return evalf,X

def true_eig(A):
    w,v = npla.eig(A)
    return w,v

print("Regual Power Iteration")
eig1,eiv1 = power_iteration(A,X)
print("Eigenvalue=",eig1,"\n","Eigenvector=\n",eiv1)
print("Inverse Power Iteration")
eig2,eiv2 = inverse_power_iteration(A,X)
print("Eigenvalue=",eig2,"\n","Eigenvector=\n",eiv2)
print("True Eigenvalues and Eigenvectors")
eig,eiv = true_eig(A)
print("Eigenvalues=",eig,"\n","Eigenvectors=\n",eiv)
