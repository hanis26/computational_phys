import numpy as np
def g(x):
    return np.sin(x)
def g_prime(x):
    return np.cos(x)

def newton(xold,kmax=200,tol=1.e-8):
    for k in range(1,kmax):
        xnew=xold-g(xold)/g_prime(xold)
        xdiff=xnew-xold
        print("{0:2d} {1:1.16f} {2:1.16f}".format(k,xnew,xdiff))

        if abs(xdiff/xnew)<tol:
            break

        xold=xnew

    else: 
        xnew=None

        return xnew
    
print(newton(3))
print(np.sin(3))
