
import math
def expapprox(x):
    n=0
    summ=0
    
    while True:
        oldsum=summ 
        term=x**n/math.factorial(n)
        summ +=term
        if math.isclose(oldsum,summ,abs_tol=0.000000001):
            break
        n+=1
    return summ,n

print (expapprox(10))
print (math.exp(10))



import math 

print ('h')
def expapprox2(x):
    n=1
    summ=1
    term =1
    while True:
        oldsum=summ 
        term=term*(x/n)
        summ +=term
        if math.isclose(oldsum,summ,abs_tol=0.000000001):
            break
        n+=1
    return summ,n

print (expapprox2(-20))
print (math.exp(-20))

