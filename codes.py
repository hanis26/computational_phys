import sympy as sp

def legendre_polynomial(n):
    x = sp.Symbol('x')
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return ((2*n-1)*x*legendre_polynomial(n-1)-(n-1)*legendre_polynomial(n-2))/n

# Test the function for n=3
print(legendre_polynomial(3))


'''LU decomp'''

import numpy as np

def lu_decomposition(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    for k in range(n-1):
        for i in range(k+1, n):
            L[i,k] = U[i,k]/U[k,k]
            U[i,k:n] = U[i,k:n] - L[i,k]*U[k,k:n]
    return L, U

# Test the function for a 3x3 matrix A
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
L, U = lu_decomposition(A)
print("A:\n", A)
print("L:\n", L)
print("U:\n", U)


'''Guass elim w partial pivoting '''

import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    x = np.zeros(n)
    # Forward elimination
    for k in range(n-1):
        # Partial pivoting
        i_max = np.argmax(abs(A[k:n,k])) + k
        if A[i_max,k] == 0:
            raise ValueError("Matrix is singular")
        A[[k,i_max]] = A[[i_max,k]]
        b[[k,i_max]] = b[[i_max,k]]
        # Elimination
        for i in range(k+1, n):
            factor = A[i,k]/A[k,k]
            A[i,k:n] = A[i,k:n] - factor*A[k,k:n]
            b[i] = b[i] - factor*b[k]
    # Backward substitution
    x[n-1] = b[n-1]/A[n-1,n-1]
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - np.dot(A[i,i+1:n], x[i+1:n]))/A[i,i]
    return x

# Test the function for a 3x3 system
A = np.array([[1, 2, -1], [2, 1, -2], [-3, 1, 1]])
b = np.array([3, 3, -6])
x = gaussian_elimination(A, b)
print("x:", x)


'''guass elim'''

import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    # Forward elimination
    for k in range(n-1):
        # Elimination
        for i in range(k+1, n):
            factor = A[i,k]/A[k,k]
            A[i,k:n] = A[i,k:n] - factor*A[k,k:n]
            b[i] = b[i] - factor*b[k]
    # Backward substitution
    x = np.zeros(n)
    x[n-1] = b[n-1]/A[n-1,n-1]
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - np.dot(A[i,i+1:n], x[i+1:n]))/A[i,i]
    return x

# Test the function for a 3x3 system
A = np.array([[1, 2, -1], [2, 1, -2], [-3, 1, 1]])
b = np.array([3, 3, -6])
x = gaussian_elimination(A, b)
print("x:", x)

'''forward and back sub'''
import numpy as np

def forward_substitution(L, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - np.dot(L[i,:i], x[:i]))/L[i,i]
    return x

def backward_substitution(U, b):
    n = len(b)
    x = np.zeros(n)
    x[n-1] = b[n-1]/U[n-1,n-1]
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - np.dot(U[i,i+1:n], x[i+1:n]))/U[i,i]
    return x

'''eigen vectors, eigenvalues'''

import numpy as np

# Define a square matrix A
A = np.array([[1, 2], [2, 1]])

# Find the eigenvalues and eigenvectors of A
eigenvalues, eigenvectors = np.linalg.eig(A)

# Print the eigenvalues and eigenvectors
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)

