# import numpy as np


y="4799 2739 8713 6272"
stripped_y=[]

for i in range(len(y)):
    if y[i]==" ":
        continue
    else:
        stripped_y.append(int(y[i]))
    
print(stripped_y)

reversed_str=reversed(stripped_y)
print(reversed_str)

print(range(len(stripped_y),0,-1))

for i in range (len(stripped_y),0,-1):
    reversed_str.append(stripped_y[i])




#2.7.3

from math import factorial

x = [i for i in range(1,100)]
for i in range(len(x)):
    sum=0
    fact=str(factorial(x[i]))
    for j in range(len(fact)+1,1,-1):
        sum = int(fact[j])+(fact[j-1])

    if sum%x[i]!=0:
        print(x)
        break


#2.7.4

a=[1,2,3]
b=[4,5,6]
c=[7,8,9]
  

def cross(x,y):
    cross=[]
    cross.append((x[1]*y[2])-(x[2]*y[1]))
    cross.append((x[2]*y[0])-(x[0]*y[2]))
    cross.append((x[0]*y[1])-(x[1]*y[0]))

    return cross
print(cross(a,b))

def dotproduct(x,y):
    dot=(x[0]*y[0])+(x[1]*y[1])+(x[2]*y[2])
    return dot

print(dotproduct(a,cross(b,c)))
print(cross(a,cross(b,c)))





    

    