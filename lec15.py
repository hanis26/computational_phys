import cmath

def k_analytic(n, x):
    T_L = (n + 1/2) * x**2
    return T_L

def psi(x, k, K):
    if x < 0:
        return cmath.exp(1j*k*x) + (1-1j*(K/k))/(1+1j*(K/k) + 1e-16)*cmath.exp(-1j*k*x)
    else:
        return 2/(1+(1j*K/k + 1e-16))*cmath.exp(-K*x)

def kinetic(psi_of_x, x, k, K, h=0.005):
    psiold1 = psi_of_x(x, k, K)
    psip1 = psi(x+h, k, K)
    psim1 = psi(x-h, k, K)
    lapl1 = (psip1 + psim1 - 2.*psiold1)/h**2
    kin1 = -0.5*lapl1/psiold1.real
    return kin1.real

print(kinetic(psi, -15, 2, 4)) 