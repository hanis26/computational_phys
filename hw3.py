import random as rand
import numpy as np
import numpy.linalg as la

# problem 4
def upper_lower(a):
    n = a.shape[0]
    L = np.identity(n)
    U = np.copy(a)

    for j in range(n - 1):
        for i in range(j + 1, n):
            coeff = U[i, j] / U[j, j]
            U[i, j:] = U[i, j:] - coeff * U[j, j:]
            L[i, j] = coeff

    return L, U


def euclidean_norm(a):
    n = a.shape[0]
    norm = 0
    for i in range(n):
        for j in range(n):
            norm = norm + (a[i, j] ** 2)

    euc_norm = np.sqrt(norm)
    return euc_norm


def infinity_norm(a):
    n = a.shape[0]
    norms = []
    for i in range(n):
        norms.append(np.sum(np.abs(a[i, :])))
    inf_norm = max(norms)
    return inf_norm


A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
L, U = upper_lower(A)
print("L:")
print(L)
print("U:")
print(U)
print("euclidean norm:", euclidean_norm(A))
print("infinity norm:", infinity_norm(A))

# problem 6

C = np.eye(2)
k = 1.0
k1_1 = 1.0
k1_2 = 1.0
a0 = 0.0
a1 = 0.0
while k < 101 and k1_1 < 1.5 and k1_2 < 1.5 and a0 < 0.001 and a1 < 0.001:
    C[0, 0] = rand.random()
    C[0, 1] = rand.random()
    C[1, 0] = rand.random()
    C[1, 1] = rand.random()
    k = la.norm(C) * la.norm(la.inv(C))
    C_tr = np.transpose(C)
    left_eigval, left_eigvector = la.eig(C_tr)
    eigval, eigvector = la.eig(C)
    k1_1 = 1 / abs(eigval[0] * left_eigval[0])
    k2_2 = 1 / abs(eigval[1] * left_eigval[1])
    a0 = abs(eigval[0] - left_eigval[0])
    a1 = abs(eigval[1] - left_eigval[1])

print("sensitive matrix:")
print(C)

"""explanation given in PDF"""

# problem 7


def fwd_sub(L, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]
    return x


def back_sub(U, b):
    n = len(b)
    x = np.zeros(n)
    x[n - 1] = b[n - 1] / U[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - np.dot(U[i, i + 1 : n], x[i + 1 : n])) / U[i, i]
    return x


def lu_inverse(A):

    n = A.shape[0]
    e = np.eye(n)

    L, U = upper_lower(A)
    ys = []
    xs = []
    for i in range(n):
        ys.append(fwd_sub(L, e[:, i]))
        xs.append(back_sub(U, ys[i]))
    xs = np.transpose(xs)
    return xs


B = np.array([[1, 2], [3, 4]])
inv = lu_inverse(B)
print("inverse:")
print(inv)
print("numpy inverse:")
print(la.inv(B))


def lu_determinant(A):
    n = A.shape[0]
    L, U = upper_lower(A)
    det = 1
    for i in range(n):
        det = det * U[i, i]
    return det


print("determinant:", lu_determinant(B))
print("numpy determinant:")
print(la.det(B))

# problem 9


def lu_pivot(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.copy(A)
    p = 0  # number of row interchanges
    for j in range(n - 1):
        k = np.argmax(np.abs(U[j:, j])) + j
        if k != j:
            p = p + 1
            U[[j, k], :] = U[[k, j], :]

        for i in range(j + 1, n):
            coeff = U[i, j] / U[j, j]
            U[i] = U[i] - coeff * U[j, j:]
            L[i, j] = coeff

    det = (-1) ** p
    for i in range(n):
        det *= U[i, i]

    return L, U, det


# make sure to enter matrix entries as floating point values and not as int
B = np.array([[1.0, 2.0], [3.0, 4.0]])
L, U, det = lu_pivot(B)
print("U:")
print(U)
print("L:")
print(L)
print("det:", det)
