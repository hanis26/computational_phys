import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, exp, pi


def y(x):
    return sin(2 * x)


def coeff(n):
    x = []
    for j in range(n):
        x.append((2 * pi * j) / n)
    m = n // 2
    a = []
    b = []
    for k in range(m):
        for j in range(n):
            a_k = (2 / n) * y(x[j]) * cos(k * x[j])
            b_k = (2 / n) * y(x[j]) * sin((k + 1) * x[j])
        a.append(a_k)
        b.append(b_k)

    return a, b


def interpol(n, x):
    a, b = coeff(n)
    m = n // 2
    p = 0.5 * a[0]
    for k in range(1, len(a)):
        p = p + a[k] * cos(k * x) + b[k] * sin(k * x)
    return p


xs = np.linspace(1, 100, 1)

f = [y(x) for x in xs]
p7 = [interpol(7, x) for x in xs]
p9 = [interpol(9, x) for x in xs]

plt.figure()
plt.plot(xs, f, "r")
plt.figure(xs, p7, "g")
plt.figure(xs, p9, "b")
plt.show()
plt.close()
