#question 1
for i in range (41):
    if i%2!=0:
        print(i)
    else:
        continue

#question 2
import numpy as np
rho=np.zeros(2000)
rho[874]=0.3
rho[1135]=0.7

#question 3(come back to this) 
def dec_to_bin(x):
    binarylist=[]
    intermediate=[]
    if x==0:
        binarylist.append(0)
    
    while x>0:
        y=x%2
        binarylist.append(y)
        x=x//2
    binarylist.reverse()
    return binarylist


print(dec_to_bin(8))

#problem 6 (come back to this)
def f(x):
    return (1/(x**3+1))

def reimannsumm(a,b,n=10):
    l=(b-a)
    stepsize=l/n
    x=[(a+stepsize*i-a*i)/2 for i in range(n)]
    result=0
    for i in range(n):
        result=result+f(x[i])*stepsize
    return result

print(reimannsumm(0,10,50))
#problem 7 

#part a
def f(x,y):
    return(x**2)*y

def euler(x0,y0,h):
    x=[]
    for j in range(100):
        x.append(x0+h*j)
        if x[j]==1:
            break

    y=[y0]

    for i in range(0,len(x),1):
        y.append(y[i-1]+h*f(x[i-1],y[i-1]))
    y.remove(1)
    return x,y

print(euler(0,1,0.5))
print(euler(0,1,0.1))

#part b 
import matplotlib.pyplot as plt
x,y=euler(0,1,0.0001)

plt.figure()
plt.plot(x,y,'r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.close()
