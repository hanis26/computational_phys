import numpy as np
from math import sqrt, pi, exp
import numpy.linalg as la
import matplotlib.pyplot as plt

# problem 1
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


A = np.array([[1.0, 2.0], [3.0, 4.0]])
L, U = upper_lower(A)

print("upper triangular:")
print(U)
print("lower triangular:")
print(L)

# problem 2

# part a
F = np.array([[1, 1], [1, 0]])

fibonacci = [F[0, 0]]
F_multiple = F
for i in range(10):
    F_multiple = la.matrix_power(F, i + 2)
    fibonacci.append(abs(F_multiple[0, 0]))

print(fibonacci)

# part b

eigval, eigvector = la.eig(F)
print("eigenvalues:")
print(eigval)
print("eigenvectors:")
print(eigvector)

# part c

fibonacci = [F[0, 0]]
F_multiple = F
for i in range(1099):
    F_multiple = la.matrix_power(F, i + 2)
    fibonacci.append(abs(F_multiple[0, 0]))

print(abs(fibonacci[1099]))

# problem 3
def f(x):
    return x**2


def integration(a, b, n):
    delta_x = abs(b - a) / n

    xs = [
        a + i * delta_x + (((a + delta_x * i) - (a + delta_x * (i - 1))) / 2)
        for i in range(1, n)
    ]
    integral = 0
    for x in xs:
        integral = integral + f(x) * delta_x

    return integral


def exact_integral(a, b):
    I = (b**3 / 3) - (a**3 / 3)
    return I


print("n=20:", integration(0, 10, 20))
print("n=100:", integration(0, 10, 100))
print("n=100000:", integration(0, 10, 100000))
print("exact integral solution:", exact_integral(0, 10))


def guassian(x, mu, sigma):
    g = (1 / (sigma * sqrt(2 * pi))) * exp(-((x - mu) ** 2) / (2 * sigma**2))
    return g


xs = np.linspace(-10, 10, 1000)


g05 = [guassian(x, 0, 0.5) for x in xs]
g1 = [guassian(x, 0, 1.0) for x in xs]
g15 = [guassian(x, 0, 1.5) for x in xs]

plt.figure()
plt.plot(xs, g05)
plt.title("sigma=0.5")
plt.show()
plt.plot(xs, g1)
plt.title("sigma=1")
plt.show()
plt.plot(xs, g15)
plt.title("sigma=1.5")
plt.show()

# part b


def integration_g05(a, b, n, mu, sigma):
    delta_x = abs(b - a) / n

    xs = [
        a + i * delta_x + (((a + delta_x * i) - (a + delta_x * (i - 1))) / 2)
        for i in range(1, n)
    ]
    integral = 0
    for x in xs:
        integral = integral + guassian(x, mu, sigma) * delta_x

    return integral


print("area for guassian with sigma=0.5:", integration_g05(-10, 10, 1000, 0, 0.5))
print("area for guassian with sigma=1.0:", integration_g05(-10, 10, 1000, 0, 1.0))
print("area for guassian with sigma=1.5:", integration_g05(-10, 10, 1000, 0, 1.5))

# part c


def derivative(x, mu, sigma, h):
    dg = (guassian(x + h, mu, sigma) - guassian(x - h, mu, sigma)) / 2 * h
    return dg


dg05 = [derivative(x, 0, 0.5, 0.00001) for x in xs]
dg1 = [derivative(x, 0, 1.0, 0.00001) for x in xs]
dg15 = [derivative(x, 0, 1.5, 0.00001) for x in xs]

plt.figure()
plt.plot(xs, dg05)
plt.title("derivative sigma=0.5")
plt.show()
plt.plot(xs, dg1)
plt.title("derivative sigma=1.0")
plt.show()
plt.plot(xs, dg15)
plt.title("derivative sigma=1.5")
plt.show()
# the value of h can be chosen as the same or smaller than the increment in x values with a grid of 1000 points


# problem 5
# part c


# problem 6
def f(x):
    return exp(x - sqrt(x))


def df(x, h=0.1):
    dg = (f(x + h) - f(x - h)) / 2 * h
    return dg


def newton(x0, x1, n):
    for k in range(n):
        x1 = x0 - (f(x0) / df(x0))
