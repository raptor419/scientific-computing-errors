
"""
@author: harsh
"""
from __future__ import division
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
import scipy.linalg as spla

A = np.array([[2,3,2],[10,3,4],[3,6,1]])
X = np.array([[2],[3],[1]])
X = np.random.random(X.shape)

def rayleigh_quotient_iteration(A,eiv):
    eps = 10**-14;
    n,m = A.shape;
    I = np.eye(n,m);
    
    X = eiv / npla.norm(eiv)
    eig = np.dot(X.T,np.dot(A,X))
    hist = [eig[0][0],]
    
    while ((len(hist)==1) or abs(hist[-2]-hist[-1])> eps ):
        X = np.dot(A,X)
        X = X / npla.norm(X)
        eig = np.dot(X.T,np.dot(A,X))
        hist += [eig[0][0],]
    
    return eig[0][0],X,hist

def rayleigh_inverse_iteration(A,eiv,eig=None):
    eps = 10**-9;
    n,m = A.shape;
    I = np.eye(n,m);
    
    X = eiv / npla.norm(eiv)
    if eig == None:
        eig = np.dot(X.T,np.dot(A,X))
        
    hist = [npla.norm(np.dot(A,X) - eig*X)/npla.norm(np.dot(A,X)),]
    eig_hist = [eig,]
    while ( hist[-1]> eps ):
        # print(hist)
        Y = A - eig*I
        # print(Y.shape,X.shape)
        X = npla.solve(Y,X)
        X = X / npla.norm(X)
        eig = np.dot(X.T,np.dot(A,X))
        eig_hist += [eig,]
        hist += [npla.norm(np.dot(A,X) - eig*X)/npla.norm(np.dot(A,X)),]
    
    return eig[0][0],X,eig_hist

print("True Eigenvalues and Eigenvectors")
eig,eiv= npla.eig(A)
print("Eigenvalues=",eig,"\n","Eigenvectors=\n",eiv)
neig,neiv = eig[0],eiv.T[0]
print("Largest Eigenvalues=",neig,"\n","Largest Eigenvector=\n",neiv)


print("Rayleigh Quotient Iteration")
eig2,eiv2,hist = rayleigh_quotient_iteration(A,X)
print("Eigenvalue=",eig2,"\n","Eigenvector=\n",eiv2)

print("Rate of Convergence")
histdif = np.log10(np.abs(eig2-hist[:-2]))
# print("Eigenvalue diff",histdif)
print("Rate of convergence",abs(hist[-1]-eig2)/abs(hist[-2]-eig2))
#order = np.log(abs(hist[-1]-hist[-2])/abs(hist[-2]-hist[-3]))/np.log(abs(hist[-2]-hist[-3])/abs(hist[-3]-hist[-4]))
#print("Order of convergance",order)

xspace = np.array(list(range(len(histdif))))
plt.title("Convergance Rate")
plt.plot(xspace, histdif,marker='',linestyle='-')
plt.xlabel('k')
plt.ylabel('delta eig log10')
plt.xlim([-1, max(xspace)+1])
plt.ylim([min(histdif)-1,max(histdif)+1])
plt.savefig('problem3.jpg')
plt.show()
