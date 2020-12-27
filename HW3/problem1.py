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

def cholesky(A):
    n = len(A)
    L = [[0.0] * n for i in range(n)]
    # choleskyfactorization from resources
    for i in range(n):
        for k in range(i+1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))
            if (i == k):
                L[i][k] = (abs(A[i][i] - tmp_sum))**(1/2)
            else:
                L[i][k] = (1.0 / L[k][k] * (A[i][k] - tmp_sum))
    return np.array(L)

def QR_Modified(A):
    m,n = A.shape[0],A.shape[1]
    q,r = np.zeros((m,n)),np.zeros((n,n))
    # qr modified from resources
    for k in range(n):
        r[k][k] = npla.norm(A[:,k],2)
        if(r[k][k]==0):
            break
        q[:,k] = A[:,k]/r[k][k]
        for j in range(k+1,n):
            r[k][j] = np.dot(q[:,k].T,A[:,j])
            A[:,j] = A[:,j] - r[k][j]*q[:,k]
    return q,r

def QR_Classic(A):
    m,n = A.shape[0],A.shape[1]
    q,r = np.zeros((m,n)),np.zeros((n,n))
    # qr from resources
    for k in range(n):
        q[:,k] = A[:,k]
        for j in range(k-1):
            r[j][k] = np.dot(q[:,j].T,A[:,k])
            q[:,k] = q[:,k] - q[:,j]*r[j][k]
        r[k][k] = npla.norm(q[:,k],2)
        if(r[k][k] == 0):
            break
        q[:,k] = q[:,k]/r[k][k]
    return q,r
    
def QR_Classic_Twice(A):
    q,r = QR_Classic(A)
    q,r = QR_Classic(q)
    return q,r

def QR_Householder(A):
    q,r = npla.qr(A)
    return q,r

def QR_Normal(A):
    L = cholesky(np.matmul(A.T,A))
    # P,L,U = spla.lu(A)
    Li = npla.inv(L)
    q = np.matmul(A,Li)
    return q,None

def QR(f,A):
    return f(A)

def quality(Q):
    return -np.log10(npla.norm(np.eye(Q.shape[1])-np.matmul(Q.T,Q)))


method = [QR_Classic,QR_Modified,QR_Classic_Twice,QR_Householder,QR_Normal]
loss = [list() for i in range(5)]
legend = ["Classic GS","Modified GS","Classic GS Twice","Householder","Normal"]
for n in range(2,13):
    # H = hilbert(n)
    for i in range(5):
        H = hilbert(n)
        q,r = QR(method[i],H)
        loss[i].append(quality(q))

xspace = np.array(list(range(2,13)))

same = True

a = list(range(3))
b = list(range(4))
c = list(range(5))

#edit question as needed
question = c

for i in c:
    if not same: plt.subplot(5, 1, i+1)
    plt.title(legend[i]+'Loss of orth vs Hilbert Order')
    plt.plot(xspace, loss[i],marker='',label=legend[i],linestyle='-')
    plt.xlabel('n')
    plt.ylabel('Loss of orth')
    
if same:
    plt.title('Loss of orth vs Hilbert Order')
    plt.legend()
plt.tight_layout() 
plt.savefig('problem1.jpg')
plt.show()

    
    
    
    
    