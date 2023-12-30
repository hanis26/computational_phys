def str_to_morse(input):
    morse = {
        "A": ".-",
        "B": "-...",
        "C": "-.-.",
        "D": "-..",
        "E": ".",
        "F": "..-.",
        "G": "--.",
        "H": "....",
        "I": "..",
        "J": ".---",
        "K": "-.-",
        "L": ".-..",
        "M": "--",
        "N": "-.",
        "O": "---",
        "P": ".--.",
        "Q": "--.-",
        "R": ".-.",
        "S": "...",
        "T": "-",
        "U": "..-",
        "V": "...-",
        "W": ".--",
        "X": "-..-",
        "Y": "-.--",
        "Z": "--..",
        "1": ".----",
        "2": "..---",
        "3": "...--",
        "4": "....-",
        "5": ".....",
        "6": "-....",
        "7": "--...",
        "8": "---..",
        "9": "----.",
        "0": "-----",
        " ": "/",
    }
    output = []
    for i in input.upper():
        if i in morse:
            output.append(morse.get(i))
            output.append(" ")
    return " ".join(output)


def morse_to_str(input):
    morse = {
        "A": ".-",
        "B": "-...",
        "C": "-.-.",
        "D": "-..",
        "E": ".",
        "F": "..-.",
        "G": "--.",
        "H": "....",
        "I": "..",
        "J": ".---",
        "K": "-.-",
        "L": ".-..",
        "M": "--",
        "N": "-.",
        "O": "---",
        "P": ".--.",
        "Q": "--.-",
        "R": ".-.",
        "S": "...",
        "T": "-",
        "U": "..-",
        "V": "...-",
        "W": ".--",
        "X": "-..-",
        "Y": "-.--",
        "Z": "--..",
        "1": ".----",
        "2": "..---",
        "3": "...--",
        "4": "....-",
        "5": ".....",
        "6": "-....",
        "7": "--...",
        "8": "---..",
        "9": "----.",
        "0": "-----",
        " ": "/",
    }

    reversed_morse = dict()
    for key in morse:
        val = morse[key]
        reversed_morse[val] = key
    output = []
    for word in input.split("/"):
        output_word = []
        for i in word.split(" "):
            if i in reversed_morse:
                output.append(reversed_morse.get(i))

        output.append(" ")
    return "".join(output).lower()


morse_code = str_to_morse(
    input()
)  # takes input from user to then convert into morse and then convert back into english
print(morse_code)
print(morse_to_str(morse_code))


# problem 2 part b
import numpy as np
from math import exp, sin, cos
import matplotlib.pyplot as plt


def f(x):
    return exp(sin(2 * x))


def fdprime(x):
    return 2 * exp(sin(2 * x)) * (-2 * sin(2 * x) + cos(4 * x) + 1)


def cal_cd(f, x, h):
    return 4 * ((f(x + (h / 2)) + f(x - (h / 2)) - 2 * f(x)) / (h**2))


hs = [10 ** (-i) for i in range(1, 11)]

x = 0.5
b = fdprime(x)

cds = []
for h in hs:
    cds.append(abs(cal_cd(f, x, h) - b))

rowf = "{0:1.0e} {1:1.16f}"
print("h       abs. error in cd")
for h, cd in zip(hs, cds):
    print(rowf.format(h, cd))
plt.figure()

plt.plot(hs, cds, "r")
plt.xlabel("h")
plt.ylabel("error (central difference)")
plt.show()
plt.close()


# problem 5 b
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, factorial, exp


def hermite(n, x):
    val0 = 1.0
    val1 = 2 * x
    for j in range(1, n):
        val2 = 2 * x * val1 - 2 * j * val0
        val0, val1 = val1, val2
    dval2 = 2 * n * val0
    return val2


xs = np.arange(-10, 10, 0.001)


def psiqho(x, n):
    momohbar = 1.0
    al = 1.0
    psival = momohbar**0.25 * exp(-0.5 * al * momohbar * x**2)
    psival *= hermite(n, np.sqrt(momohbar) * x)
    psival /= np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))
    return psival


def classical(x, E=1, m=1, omega=1):
    A = np.sqrt(2 * E / m) / omega
    return np.exp(-(x**2) / A**2) / (np.pi ** (1 / 2) * A)


h3_sq = [(psiqho(x, 3)) ** 2 for x in xs]
h10_sq = [(psiqho(x, 10)) ** 2 for x in xs]
h20_sq = [(psiqho(x, 20)) ** 2 for x in xs]
h150_sq = [(psiqho(x, 150)) ** 2 for x in xs]
classical1 = [classical(x) for x in xs]


plt.figure()
plt.plot(xs, classical1)
plt.title("classical probability density")
plt.show()

plt.plot(xs, h3_sq, label="n=3")
plt.plot(xs, h10_sq, label="n=10")
plt.plot(xs, h20_sq, label="n=20")
plt.plot(xs, h150_sq, label="n=150")

plt.legend()
plt.xlabel("x")
plt.title("Square of Wavefunctions for 1D Harmonic Oscillator")

plt.show()

plt.close()


# problem 6
import numpy as np
from math import sqrt, pi, factorial, exp

import numpy as np


def hermite(n, x):
    val0 = 1.0
    val1 = 2 * x
    for j in range(1, n):
        val2 = 2 * x * val1 - 2 * j * val0
        val0, val1 = val1, val2
    dval2 = 2 * n * val0
    return val2


def psiqho(x, nametoval):
    nx = nametoval["nx"]
    ny = nametoval["ny"]
    nz = nametoval["nz"]
    momohbar = nametoval["momohbar"]
    al = nametoval["al"]
    psix = momohbar**0.25 * exp(-0.5 * al * momohbar * x[0] ** 2)
    psiy = momohbar**0.25 * exp(-0.5 * al * momohbar * x[1] ** 2)
    psiz = momohbar**0.25 * exp(-0.5 * al * momohbar * x[2] ** 2)
    psix *= hermite(nx, sqrt(momohbar) * x[0])[0]
    psiy *= hermite(ny, sqrt(momohbar) * x[1])[0]
    psiz *= hermite(nz, sqrt(momohbar) * x[2])[0]
    psival = psix * psiy * psiz
    psival /= sqrt(
        2 ** (nx + ny + nz)
        * factorial(nx)
        * factorial(ny)
        * factorial(nz)
        * sqrt(pi) ** 3
    )
    return psival


def kinetic(psi, x, nametoval, h=0.005):
    hom = 1.0
    psiold = psi(x, nametoval)
    psixp = psi([x[0] + h, x[1], x[2]], nametoval)
    psixm = psi([x[0] - h, x[1], x[2]], nametoval)
    laplx = (psixp + psixm - 2.0 * psiold) / h**2
    psiy = psi([x[0], x[1] + h, x[2]], nametoval)
    psiy = psi([x[0], x[1] - h, x[2]], nametoval)
    laply = (psiy + psiy - 2.0 * psiold) / h**2
    psizp = psi([x[0], x[1], x[2] + h], nametoval)
    psizm = psi([x[0], x[1], x[2] - h], nametoval)
    laplz = (psizp + psizm - 2.0 * psiold) / h**2
    lapl = laplx + laply + laplz
    kin = -0.5 * hom * lapl / psiold
    return kin


# problem 7


import itertools
import math


def find_triples(n_x, n_y, n_z):
    tuples = list(itertools.permutations([n_x, n_y, n_z]))
    distinct_tuples = []
    for tpl in tuples:
        if tpl not in distinct_tuples:
            distinct_tuples.append(tpl)
    return distinct_tuples


def cardinal_eig(n):
    eigenvectors = []
    for i in range(1, n):
        for j in range(1, n):
            for k in range(1, n):
                if i**2 + j**2 + k**2 == n:
                    eigenvectors.extend(find_triples(i, j, k))
    return eigenvectors


def x_psi(x, n_x, L):
    return math.sqrt(2 / L) * math.sin((n_x * math.pi * x) / L)


def y_psi(y, n_y, L):
    return math.sqrt(2 / L) * math.sin((n_y * math.pi * y) / L)


def z_psi(z, n_z, L):
    return math.sqrt(2 / L) * math.sin((n_z * math.pi * z) / L)


def psibox_3d(n, x, y, z, L):
    psi_in_3d = []
    for j in cardinal_eig(n):
        psi_in_3d.append(x_psi(x, j[0], L) * y_psi(y, j[1], L) * z_psi(z, j[2], L))
    return psi_in_3d


print(psibox_3d(70, 3, 6, 5, 2 * pi))


# problem 8

import cmath


def k_analytic(n, x):
    T_L = (n + 1 / 2) * x**2
    return T_L


def psi(x, k, K):
    if x < 0:
        return cmath.exp(1j * k * x) + (1 - 1j * (K / k)) / (
            1 + 1j * (K / k) + 1e-16
        ) * cmath.exp(-1j * k * x)
    else:
        return 2 / (1 + (1j * K / k + 1e-16)) * cmath.exp(-K * x)


def kinetic(psi_of_x, x, k, K, h=0.005):
    psiold1 = psi_of_x(x, k, K)
    psip1 = psi(x + h, k, K)
    psim1 = psi(x - h, k, K)
    lapl1 = (psip1 + psim1 - 2.0 * psiold1) / h**2
    kin1 = -0.5 * lapl1 / psiold1.real
    return kin1.real


print(kinetic(psi, -15, 2, 4))
