# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 09:35:24 2017

@author: Albert
"""

#Assertions
"""
Sometimes our functions might not be available for certain values splitting out an error message. 
however, if we know the error is better to explicitly detail the problem.
"""
#EX: Variance
def var(x):
    n=len(x)
    assert n>1,  "Sample size must be larger than 1"
    return (np.sum(x-x.mean**2))/(n-1)

var([9])



#DECORATORS AND DESCRIPTORS
"""
synctatic sugar that can be avoided but quite popular.
For example we might create different functions that can have the same assertion problem.
We could just in each function copy the assert line. However this is repetition and we try to avoid it.
It is better to use a decorator.
"""

#Example
import numpy as np

def check_nonneg(func):
    def safe_function(x):
        assert x >= 0, "Argument must be nonnegative"
        return func(x)
    return safe_function

def f(x):
    return np.log(np.log(x))

def g(x):
    return np.sqrt(42 * x)

f = check_nonneg(f)
g = check_nonneg(g)

#Decorators help replace the lines of check non-negat and show clearer the assert line:
@check_nonneg
def f(x):
    return np.log(np.log(x))

@check_nonneg
def g(x):
    return np.sqrt(42 * x)


#Example of descriptors
class Car:

    def __init__(self, miles=1000):
        self._miles = miles
        self._kms = miles * 1.61

    def set_miles(self, value):
        self._miles = value
        self._kms = value * 1.61

    def set_kms(self, value):
        self._kms = value
        self._miles = value / 1.61

    def get_miles(self):
        return self._miles

    def get_kms(self):
        return self._kms

    miles = property(get_miles, set_miles)
    kms = property(get_kms, set_kms)
    
"""
How it Works
The names _miles and _kms are arbitrary names we are using to store the values of the variables
The objects miles and kms are properties, a common kind of descriptor
The methods get_miles, set_miles, get_kms and set_kms define what happens when you get (i.e. access) or set (bind) these variables
So-called “getter” and “setter” methods
The builtin Python function property takes getter and setter methods and creates a property
For example, after car is created as an instance of Car, the object car.miles is a property
Being a property, when we set its value via car.miles = 6000 its setter method is triggered — in this case set_miles
"""


#Another version with Decorators to set up Properties
class Car:

    def __init__(self, miles=1000):
        self._miles = miles
        self._kms = miles * 1.61

    @property
    def miles(self):
        return self._miles

    @property
    def kms(self):
        return self._kms

    @miles.setter
    def miles(self, value):
        self._miles = value
        self._kms = value * 1.61

    @kms.setter
    def kms(self, value):
        self._kms = value
        self._miles = value / 1.61
        


#Generators expression
# Generators are a kind of iteration. the easiest way to build generators expressions
#is using just like a list comprehension, but with round brackets.
#EXAMPLE
singular = ('dog', 'cat', 'bird')
type(singular)
plural = [string + 's' for string in singular]
plural
type(plural)
#and the generator expession
singular = ('dog', 'cat', 'bird')
plural = (string + 's' for string in singular)
type(plural)
next(plural)

#Example2: sum() can be called on iterators
sum(x * x for x in range(10))





#Generators functions
#The most flexible way to create generator objects is to use generator functions:
#Example 1:
def f():
    yield 'start'
    yield 'middle'
    yield 'end'
    
type(f)
gen=f()
next(gen)



#Example 2
def g(x):
    while x < 100:
        yield x
        x = x * x

gen = g(2)  
type(gen) 
next(gen)    
        
        
#Generator function for a Binomial
import random
def f(n):
    i = 1
    while i <= n:
        yield random.uniform(0, 1) < 0.5
        i += 1
        
n = 10000000
draws = f(n)
draws        






# Recursive function Calls
"""
   Recursive functions are functions that call itself.
"""
   
#EXAMPLE 1: Xt+1 = 2Xt
def x_loop(t):
   x=1
   for i in range(t):
       x=2*x
   return

#and the recursive solution: (more efficient)
def x(t):
    if t==0:
        return 1
    else:
        return 2*x(t-1)
    

#EXAMPLE 2: Fibonacchi sequence in recursive form
def x(t):
    if t == 0:
        return 0
    if t == 1:
        return 1
    else:
        return x(t-1) + x(t-2)

print([x(i) for i in range(10)])    
    
    
