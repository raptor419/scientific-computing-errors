
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

def modified_qr_iteration(A):
    eps = 10**-15
    eigi,eigf = None,None
    n,m = A.shape
    I = np.eye(n,m);
    X = np.eye(n,m)
    eigf = A[n-1,m-1]
    while(eigi==None or abs(eigi-eigf)>eps):
        # print(A)
        eigi = eigf
        A = A - eigi*I
        Q,R = npla.qr(A)
        X = np.dot(X,Q)
        A = np.matmul(R,Q) + eigi*I
        eigf = A[n-1,m-1]
    A = np.diagonal(A)
    return A,X

def true_eig(A):
    w,v = npla.eig(A)
    return w,v

print("\n")
print("Applying on Matrix 1")
A = A1
print(A)
print("Modified QR Iteration")
eig2,eiv2 = modified_qr_iteration(A)
print("Eigenvalues=",eig2,"\n","Eigenvectors=\n",eiv2)
print("True Eigenvalues and Eigenvectors")
eig,eiv = true_eig(A)
print("Eigenvalues=",eig,"\n","Eigenvectors=\n",eiv)

print("\n")
print("Applying on Matrix 2")
A = A2
print(A)
print("Modified QR Iteration")
eig2,eiv2 = modified_qr_iteration(A)
print("Eigenvalues=",eig2,"\n","Eigenvectors=\n",eiv2)
print("True Eigenvalues and Eigenvectors")
eig,eiv = true_eig(A)
print("Eigenvalues=",eig,"\n","Eigenvectors=\n",eiv)