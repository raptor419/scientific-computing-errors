from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npla
import scipy.linalg as spla

def makelu(k):
    ep = 10**((1)*(2*k))
    e = 10**((-1)*(2*k))
    U = np.array([[e,1],[0,1-ep]])
    L = np.array([[1,0],[ep,1]])
    x = np.array([[1],[1]])
    b = np.array([[1+e],[2]])
    return L,U,x,b

def makeA(k):
    e = 10**((-1)*(2*k))
    A = np.array([[e,1],[1,1]])
    B = np.array([[1+e],[2]])
    x = np.array([[1],[1]])
    return A,B,x
    

def lusolve(L,U,B):
    # print(L,U,B)
    y = spla.solve_triangular(L,B,lower=True)
    x = spla.solve_triangular(U,y)
    return x

def irsolve(A,B):
    x_hat = npla.solve(A,B)
    # print("in_hat",x_hat)
    for i in range(1):
        r = B - np.matmul(A,x_hat)
        y = npla.solve(A,r)
        x_hat = x_hat + y
    return x_hat


def irsolve2(A,b):
    L, U, _, _ = makelu(k)
    y = npla.solve(L,b)
    x = npla.solve(U,y)
    xi = x
    for i in range(1):
        r = np.subtract(b,np.matmul(A,x))
        z = npla.solve(L,r)
        s = npla.solve(U,z)
        xi = xi + s
    return xi




e = np.array([],dtype=np.double)
eir = np.array([],dtype=np.double)


n=10
for k in range(1,n):
    L,U,x,b = makelu(k)
    x_hat = lusolve(L,U,b)
    print("xhat1",x_hat)
    e=np.append(e,npla.norm(x - x_hat, ord=None))
    A,B,x = makeA(k)
    x_hat = irsolve(A,B)
    print("xhat2",x_hat)
    eir=np.append(eir,npla.norm(x - x_hat,ord=None))

xspace = np.array(list(range(1,n)))

same = True

if not same: plt.subplot(1, 1, 1)
plt.title('Log Absolute Error inf norm vs Epsilon')
plt.plot(xspace, np.log(e),marker='',label='GE /wo pivot',linestyle='-')
plt.plot(xspace, np.log(eir),marker='',label='GE /w pivot Itr refined',linestyle='-')
plt.xlabel('k')
plt.ylabel('norm')

if same:
    plt.ylabel('norm')
    plt.legend()
plt.tight_layout() 
plt.savefig('problem2.jpg')
plt.show()
    




