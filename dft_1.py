from math import exp, sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return sin(x)**2


def coeff(m):
    n = 2*m
    x = np.array([2*np.pi*i/n for i in range(n)])
    a = []
    b = []

    for j in range(m):
        a_k = 0
        b_k = 0
        for i in range(n):
            a_k = a_k+f(x[i])*cos(j*(2*pi*i/n))
            b_k = b_k+f(x[i])*sin(j*(2*pi*i/n))
        a.append(a_k/m)
        b.append(b_k/m)
    return [a, b]


def p(m, x):
    n = 2*m
    a, b = coeff(n)
    p = 0.5*a[0] + a[m-1]*cos((m-1)*x)/2
    for k in range(1, m-1):
        p = p+a[k]*cos(k*x) + b[k-1]*sin(k*x)
    return p


xs = np.arange(1, 100, 1)
ps = [p(500, x) for x in xs]
fs = [f(x) for x in xs]

plt.figure()
plt.plot(xs, ps, 'r', "fourier")
plt.plot(xs, fs, 'b', "analytical")
plt.show()
plt.close()
