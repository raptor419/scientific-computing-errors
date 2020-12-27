
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

A = A2
X = np.random.random(X.shape)

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

def shifted_power_iteration(A,X,est=2):
    A = A - est*np.eye(A.shape[0],A.shape[1])
    # print(A)
    #print(inverse_power_iteration(A, X))
    #print(true_eig(As))
    eig, eiv = inverse_power_iteration(A, X)
    eig, eiv = eig+est, eiv
    return eig, eiv
    

def true_eig(A):
    w,v = npla.eig(A)
    return w,v

# print("Inverse Power Iteration")
# eig1,eiv1 = inverse_power_iteration(A,X)
# print("Eigenvalue=",eig1,"\n","Eigenvector=\n",eiv1)
print("Shifted Inverse Power Iteration")
eig2,eiv2 = shifted_power_iteration(A,X)
print("Eigenvalue=",eig2,"\n","Eigenvector=\n",eiv2)
print("True Eigenvalues and Eigenvectors")
eig,eiv = true_eig(A)
print("Eigenvalues=",eig,"\n","Eigenvectors=\n",eiv.T)
