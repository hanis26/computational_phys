#problem 1
list1 = [2,4,5,9,8,6]
def secondelement(list1):
    list2=[]
    for i in range (len(list1)):
        if (i%2==0):
            list2.append(list1[i])
        else:
            continue
        i=i+1
        
    return list2

print(secondelement(list1))

#problem 2
  #part a
x=27
seq=[]
if x==1:
    print(x)
else:
    seq.append(x)

while x!=1:

    if x%2==0:
        x=int(x/2)
        seq.append(x)

    elif x%2!=0:
        x=int(3*x+1)
        seq.append(x) 

print(seq)

  #part b 

def hailstone(x):

    seq=[]
    if x==1:
        seq.append(1)
    else:
        seq.append(x)
        
    while x!=1:

        if x%2==0:
            x=int(x/2)
            seq.append(x)

        else:
            x=int(3*x+1)
            seq.append(x)
    y=len(seq)    
    return y

collatz=[]
for i in range(1,101):
    t=hailstone(i)
    collatz.append(t)

print(collatz)

#since it returns a finite list for all the numbers 1-100, then the Collatz conjecture has been proven.


#problem 3 

# def seive(n):
#     p=[]
#     m=[]
#     multiples=[]
#     primes=[]

#     p = [i for i in range(2,n)]
#     m = [i for i in range(2,n)]

#     for i in range (len(p)):
#         multiples.append(p[i]*m[i])

#     for i in p:
#         if i in multiples:
#             continue
#         else:
#             primes.append(i)
#     return primes
            
# print(seive(10))

# def seive(n):
#     p=[]
#     m=[]
#     multiples=[]
#     primes=[]

#     p = [i for i in range(2,n)]
#     m = [i*j for j in range(2,n)]

#     for i in range (len(p)):
#         multiples.append(p[i]*m[i])

#     for i in p:
#         if i in multiples:
#             continue
#         else:
#             primes.append(i)
#     return primes
            
# print(seive(10))
list1=[1,1]+[0]*10000
p=0
for counter,item in enumerate(list1):
    if item ==0:
        p=counter
        for i in range(p*p,len(list1),p):
            list1
j=0
for i in range(2,len(list1)):
    if list1[i]==0:
        j+=1
print(j)
         

#problem 4

def factorial(y):
    fact=1
    if y==0:
        return fact
    else:
        for i in range (1,y):
            fact=fact*(i+1)
    return fact

print(factorial(4))

#problem 5 

def tetra(n,x):
    value=x
    if n==0:
        value = 1
    else:
        value=value**tetra(n-1,x)
    return value
print(tetra(4,2))

#problem 6

from math import sqrt
def f(x,nmax=100):
    for i in range(nmax):
        x = sqrt(x)
        print("square root is", x)
    for i in range(nmax):
        x = x**2
        print("square is", x)
    return x
for xin in (5., 0.5):
    xout = f(xin)
    # print(xin,xout)
print(xin, xout)

#the output we get is not the same as the input that the function is called with
#because the float can only save a certain number of significant figures,
#without any means of preserving accuracy. This causes the accuracy to be lost each time the sqrt is called
#causing a loss in accuracy for when the value is squared the same number of times. 


#problem 7
    


#part b
import math
def badformula(a,c,b):
    x_minus=((-b)-math.sqrt((b*b)-(4*a*c)))/2*a
    x_plus=((-b)+math.sqrt((b*b)-(4*a*c)))/2*a
    return x_minus,x_plus

print(badformula(1,1,1e8))

def goodformula(a,c,b):
    x_minus=((-b)-math.sqrt((b*b)-(4*a*c)))/2*a
    x_plus=c/(a*x_minus)
    return x_minus,x_plus

print(goodformula(1,1,1e8))

#discuss

#problem 8

x_1 = 1234567891234567
y_1 = 1234567891234566
result_1 = (x_1**2)-(y_1**2)

print(result_1)

x_2=1234567891234567.0
y_2=1234567891234566.0
result_2=(x_2**2)-(y_2**2)
print(result_2)

result_corrected=(x_2-y_2)*(x_2+y_2)

print(result_corrected)

#it does match because there is no catastrophic cancellation anymore because we are not 
#subtracting from a square number


#problem 9
#part a
import math
summation=0
nmaxd=0
summation_list=[0]

for i in range(1,1000000):
    summation=summation+(1/(i**2))
    summation_list.append(summation)
    if math.isclose(summation_list[i-1],summation_list[i],rel_tol=1e-10):
        break
    


print(summation)
nmaxd=i
print(nmaxd)





#part b

#1/k^2 becomes smaller than the tolerance set for the isclose function (here it is the default value of 1e-10)
#therefore, the summation stops changing significantly changing at nmaxd

#part c\
nmaxr=nmaxd*4

summ_list_1=[]

def reimannzeta(x):
    summ=0
    for i in range(x,0,-1):
        summ=summ+(1/(i**2))
        i=i+1

    return summ

print(reimannzeta(nmaxr))
print(reimannzeta(nmaxr*2))
print(reimannzeta(nmaxr*4))
print(reimannzeta(nmaxr*8))

#part d

def kahansum(xs):
    s=0.;e=0.
    for x in xs:
        temp=s
        y=x+e
        s=temp+y
        e=(temp-s)+y
    return s


listnum = [(1/i**2) for i in range(1,1000000)]

print(kahansum(listnum))


#problem 10

#part a

import numpy as np
import matplotlib.pyplot as plt

def sqwave(x):
    return np.where((x > -np.pi) & (x <= 0), -1/2, np.where((x > 0) & (x < np.pi), 1/2, 0))

def fourierseries(x, n):
    y = np.sum([np.sin(i * x) / i for i in range(1, n+1, 2)])
    return (2/np.pi) * y

x = np.arange(-np.pi, np.pi, 0.05 * np.pi)
y1 = sqwave(x)
y2 = fourierseries(x, 1)
y3 = fourierseries(x, 3)
y4 = fourierseries(x, 5)
y5 = fourierseries(x, 7)
y6 = fourierseries(x, 5000)

y2max = np.max(y2)
x2_max = x[np.argmax(y2)]
y3max = np.max(y3)
x3_max = x[np.argmax(y3)]
y4max = np.max(y4)
x4_max = x[np.argmax(y4)]
y5max = np.max(y5)
x5_max = x[np.argmax(y5)]
y6max = np.max(y6)
x6_max = x[np.argmax(y6)]

plt.plot(x, y1, color="g")
plt.plot(x, y2, color="r")
plt.plot(x, y3, color="b")
plt.plot(x, y4, color="m")
plt.plot(x, y5, color="c")
plt.plot(x, y6, color="y")

plt.grid()
plt.legend(["Square Wave", "Fourier wave, n=1", "Fourier wave, n=3", "Fourier wave, n=5", "Fourier wave, n=7", "Fourier wave, n=9"])

plt.show()


'''
As we increase the values of n from 1 to 9 in the Fourier series, the maximum value of the function gradually approaches the point x = pi. Initially, at n = 1, the oscillation amplitude is high with no destructive interference, resulting in a maximum value that is relatively higher. However, as we increase n, the amplitude decreases, approaching a value of 0.5. Eventually, for larger values of n, the maximum value comes very close to 0.5 and continues to converge towards it.'''