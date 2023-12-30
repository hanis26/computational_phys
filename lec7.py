import numpy as np
import matplotlib.pyplot as plt

#hermite polynomials
x=0.5

#def hermite(n,x):
    #h_list=[]
    #h_list.append(1)
    #h_list.append(2*x)
    #for i in range(1,n+1):
    #    h_list.append(2*x*(h_list[i+1])-2*n*h_list[i])
    
    #return h_list

#print(hermite(3,x))


def hermite_1(n,x):
    val0=1
    val1=2*x
    for j in range(1,n):
        val2=2*x*val1-2*j*val0
        val0,val1=val1,val2
    dval2=2*n*val0
    return val2,dval2

hermitevalue,hermitederivative = hermite_1(3,0.5)
print(hermitevalue)
print(hermitederivative)