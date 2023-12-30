import numpy as np
import numpy.linalg as la

L=[11,7,19,22,25]

xs=np.array(L)
# print(xs)
# print(L)

# print(len(L))
# print(xs.dtype)



# print(xs[1])

# print(np.array([1.]*4))
# print(xs.shape)

# print(np.zeros(4))
# print(np.linspace(0.0,1.0,5))


# xs=np.array(L,dtype=np.float64)

# print(xs[1])
# print(xs[1:3])
# test=xs[1:3]
# test[:]=0.0
# print(xs)

# np.copy(xs[1:3])


# P=[1,2,3,4,5]

# Larr=np.array(L)
# Parr=np.array(P)

# print(Larr*2.0)

# def f(x):
#     return x**2.0 +2.0*x

# print(f(Larr))


# LL=[[11,12,13,14],[15,16,17,18],[19,20,21,21]]

# A=np.array(LL)

# print(LL)

# print(A)

# print(A.size)

# print(A.ndim)
# print(A.shape)

# B=np.identity(3)

# B=np.array([1,4,9,7,2,8,5,6])

# print(B.shape(2,4))

# C=B.reshape(2,4)

# print(C[1,1])

# A=np.arange(1,10).reshape(3,3)

# B=np.arange(11,20).reshape(3,3)

# C=A@B

# np.transpose(A)

# A.T

# np.trace

# print(A)
# print(B)
# print(C)

a = np.array([[0.2161,0.1441],[1.2969,0.8648]])
b = np.array([0.1440,0.8642])

x = la.solve(a,b)
print(x)