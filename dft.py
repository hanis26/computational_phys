from math import exp, sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return exp(sin(4*x))

def coeff(m):
    n = 2*m
    x = np.array([2*np.pi*i/n for i in range(n)])
    a = []
    b = []

    for j in range(m):
        a_k = 0
        b_k = 0
        for i in range(n):
            a_k += (1/m)*f(x[i])*cos(j*(2*pi*i/n))
            b_k += (1/m)*f(x[i])*sin((j+1)*(2*pi*i/n))
        a.append(a_k)
        b.append(b_k)
    return a, b

def p(a, b, n, x):
    ps = []
    p = (a[0]/2) + (a[n//2]*cos(n*x)/2)
    for k in range(1, n//2):
        p += a[k]*cos(k*x) + b[k-1]*sin(k*x)
        ps.append(p)
    return ps


xs = np.arange(0, 2*np.pi, 0.01)
fs = [f(x) for x in xs]
a, b = coeff(50)
ps = p(a, b, 100, xs)

plt.plot(xs, ps, 'r')
plt.plot(xs, fs, 'b')
plt.show()


