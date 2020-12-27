from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npla
import scipy.linalg as spla

def hilbert(n):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / ((i + 1) + (j + 1) - 1)
    return H

def lusolve(A,B):
    P,L,U = spla.lu(A)
    B = np.matmul(P,B)
    y = spla.solve_triangular(L,B)
    x = spla.solve_triangular(U,y)
    return x

r = []
e = []

n = 2
while True:
    H = hilbert(n)
    x = np.ones(n)
    b = H.dot(x)
    x_hat = lusolve(H,x)
    con = npla.norm(x - x_hat, ord = np.inf)
    print(n,con)
    if(n>20):
        break
    r.append(npla.norm(b - H.dot(x_hat), ord = np.inf))
    e.append(npla.norm(x - x_hat, ord = np.inf))
    n+=1


xspace = np.array(list(range(2,n)))

same = False

if not same: plt.subplot(3, 1, 1)
plt.title('Residual inf norm vs Hilbert Order')
plt.plot(xspace, r,marker='',label='Residual inf norm',linestyle='-')
plt.xlabel('n')
plt.ylabel('norm')
if not same: plt.subplot(3, 1, 2)
plt.title('Absolute Error inf norm vs Hilbert Order')
plt.plot(xspace, e,marker='',label='Absolute Error',linestyle='-')
plt.xlabel('n')
plt.ylabel('norm')
if not same: plt.subplot(3, 1, 3)
plt.title('Log Absolute Error inf norm vs Hilbert Order')
plt.plot(xspace, np.log10(e),marker='',label='Absolute Error',linestyle='-')
plt.xlabel('n')
plt.ylabel('norm')

if same:
    plt.ylabel('p(x)')
    plt.legend()
plt.tight_layout() 
plt.savefig('problem1.jpg')
plt.show()
