import numpy

height ={'burj khalifa':828.,'one world trade center':541.3,'mercury city tower':-1.,'Q1':323.
,'cariton centre':223.,'gran torre santiago':300.,'mercury city tower':339.}
print(height['burj khalifa'])

print(height.get('burj khalifa'))

s='all you base are belong to us'


print(set(s.lower()) >= set('abcdefghijklmnopqrstuvwxyz'))


def reciprocal(x):
    try:
        ans = 1.0/x
        return ans
    except ZeroDivisionError:
        return 1.0
        print('Division by zero is not allowed.')


import math
def sinc(x):
    try:
        return math.sin(x)/x
    except ZeroDivisionError:
        return 1.0

print(sinc(1))


def powr(a,b):
    if a==b==0:
        raise ValueError('Both a and b cannot be zero!!!')
    return a**b

print(powr(0,0))

   



