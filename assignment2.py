#problem 1
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
def legendre(n,x):
    leg0=1
    leg1=x
    leg_list=[]
    leg_list.append(leg0)
    leg_list.append(leg1)
    for i in range(1,n+1):
        leg2=((2*i+1)*x*leg1-(i*leg0))/(i+1)
        leg_list.append(leg2)
        leg0,leg1=leg1,leg2

    return leg_list
list=[]
for i in range(5):
    for x in arange(1,50,1):
        list.append(legendre(i,x))


leg1=[]
leg2=[]
leg3=[]
leg4=[]
leg5=[]
for i in range(0,50):
    leg1.append(list[i])

for i in range(50,100):
    leg2.append(list[i])
for i in range(100,151):
    leg3.append(list[i])

for i in range(150,201):
    leg4.append(list[i])
for i in range(200,245):
    leg5.append(list[i])


x=[arange(1,51,1)]
plt.figure()
plt.plot(x,leg1)
plt.plot(x,leg2)
plt.plot(x,leg3)
plt.plot(x,leg4)
plt.plot(x,leg5)
plt.close()
#problem 2

