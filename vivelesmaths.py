import numpy as np

def swap(v, w):
    ex1 = np.copy(w)
    w = np.copy(v)
    v = ex1

def Gauss(A, b):
    n = len(b)
    for j in range(n-1):
        #meilleur pivot possible
        if A[j][j] != max(np.abs([A[i][j] for i in range(j, n)])):
            v = np.abs([A[i][j] for i in range(j, n)])
            k = np.where(v == max(v))
            swap(A[j], A[k])
            swap(b[j], b[k])
        #triangularisation
        for i in range(j+1, n):
            if A[i][j] != 0:
                div = A[i][j] / A[j][j]
                A[i] = A[i] - div * A[j]
                b[i] = b[i] - div * b[j]
    #calculs des zeros
    x = np.zeros(n)
    for i in np.arange(n-1, -1, -1):
        if A[i][i] == 0.:
            x[i] = b[i]
        else:
            x[i] = (b[i] - np.dot(A[i][i+1:n], x[i+1:n]))/A[i][i]
    return x

def index_max(v):
    m = v[0]
    j = 0
    for i, v_i in enumerate(v):
        if v_i > m:
            j = i
    return j

def Gauss_J(A, b):
    n = len(b)
    for j in range(n-1):
        #meilleur pivot possible
        A_i = np.abs([A[i][j] for i in range(j, n)])
        if A[j][j] != max(A_i):
            k = index_max(A_i)
            swap(A[j], A[k])
            swap(b[j], b[k])
        #triangularisation
        for i in range(j+1, n):
            if A[i][j] != 0:
                div = A[i][j] / A[j][j]
                A[i] = A[i] - div * A[j]
                b[i] = b[i] - div * b[j]
    #calculs des zeros
    x = np.zeros(n)
    for i in np.arange(n-1, -1, -1):
        if A[i][i] == 0.:
            x[i] = b[i]
        else:
            x[i] = (b[i] - np.dot(A[i][i+1:n], x[i+1:n]))/A[i][i]
    return x

