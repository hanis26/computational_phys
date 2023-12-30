#gezerlis triang.py
import numpy as np

def gauelim(inA,inbs):
    A=np.copy(inA)
    bs=np.copy(inbs)
    n=bs.size

    for j in range(n-1):
        for i in range(j+1,n):
            coeff=A[i,j]/A[j,j]
            A[i,j:]-=coeff*bs[j]

    xs=backsub(A,bs)
    return xs

if name=
