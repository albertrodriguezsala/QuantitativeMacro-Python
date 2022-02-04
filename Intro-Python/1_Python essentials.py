# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 16:11:55 2017

@author: Albert
"""

# =============================================================================
# Python essentials
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# On good coding practices:
import this

# Python is based on Object Oriented Programming:
'''
Python supports both procedural and object-oriented programming. Matlab and fortram are mainly procedural while 
C++ is procedural with OOP added on top.
   
->  The procedural paradigm works as follows: Program has a state that contain values of its variables
       Fuctions are called to act on these data, data are passed back via function calls.

-> the OOP paradigm DATA AND FUNCTIONS BUNDLED TOGETHER into objects. 
      - In the OOP functions are usually called methods.
      - In python the data and methods of an object are referred to as attributes.
      - Depending on the class of object(like float, inter, str, etc) the object will have different methods.
      
OOP is useful as for the same reason abstraction is: for recognizing and exploiting common structure.

In Python everything in memory is treated as an object. This includes not just lists, strings, etc.,
but also less obvious things, such as: functions (once they have been read into memory), modules (ditto), 
files opened for reading or writing, integers, etc.
    
In Python, an object is:
   -> a collection of data and instructions held in computer memory that consists of: 
       a type, 
       some content, 
       a unique identity, 
       methods.
'''


# =============================================================================
# # Variables and Data types:
# =============================================================================

# 1 List --------------------------------------
list_1 = [1,2,3]

y = lambda x: x**2

list_obj = [1,2,3,'a','b', y]
list_obj2 = [[1,2,3,], ['a','b']]

### Main virtue of lists: Its versatility. List of dataframes. List of list of list, dataframes, arrays, strings, etc. 

#### Indexing in python: starts in the 0!!!
list_obj[0]

list_obj2[0]
## If we want to get the 1st element:
list_obj2[0][0]
list_obj[0:3]
list_obj[-2] #to access the second last element of a list
list_obj[2:] #to access the two last elements of a list.

## lists are mutable objects:
list_obj[0] = np.ones(5)



# 2 Arrays -------------------------------------
## We use the numpy library
a = np.array([[1,2,3],[4,5,6]])

# 3 Boolean values  ----------------------------  
x= True
y = 100 < 10
type(y)
y
x+y     #In numerical expressions true=1 and false=0
x*y
#Boolean arithmetic 
bools = [True, True, False, True]
sum(bools)


# 4 Numbers --------------------------------------
# 4.1 Integers
a, b = 1, 2

# 4.2 Floats
c, d = 2.5, 10.0
#Computer distinguishes between floats and integers because floats are more informative 
#while arithmetic operations on integers are faster and more accurate.
1/2 #Normal division
1//2 #integer division

# 4.3 Complex numbers
x = complex(1,2)
y = complex (2,1)
x, y
x*y



# 5 Containers------------------------------------
# 5.1 Lists vs tuples
#We have lists and a related data type is tuples which are immutable lists:
x = ['a','b'] #list
y= ('a','b')
y = 'a','b'  #Both ways work to create a tuple.

x[0]= 20
y[0] = 20 #note that tuples aren't mutable

#Tuples and lists can be unpacked as follows:
integers = [10,20,30]
x, y, z = integers



#it also works on strings and tuples:
b='foobar'
b[-3:]


# 5.2 Dictionaries and sets 
#dictionaries are lists but ordered by names instead of numbers:
individual = {'name':'Xivito', 'age':28, 'weight':100} #name, age... are denominated the keys
individual['age']
#Sets are unordered  collections without duplicates
s1 = {'a','b'}
s2 = {'c','b'}
s1.issubset(s2)
s1.intersection(s2)


# 6. Functions -------------------------------------
#some examples
x= [0,1,2,3]

max(x)
range(max(x))
bools = True, False, True
all(bools)
any(bools)

# Functions are very flexible on Python. Any number of fuctions can be defined in a given file. 
# Functions can be defined inside other functions. any object can be passed to a function as an argument, 
# including functions. A function can  return any kind of object, including functions.


#lambda is used to create one-line functions:
f= lambda x: x**3


#Example: suppose we want to compute the integral on (0,2) of x***3
from scipy.integrate import quad
quad (lambda x: x**3, 0, 2)

#Keyword arguments
#In general we have two types of arguments: positional arguments and keyword arguments. 
# keyword arguments are created by label='argument'

plt.plot(x,'b-', label='white noise') #keyword arguments are useful for functions with many arguments (like plots)

#We can adopt keyword to our functions:
    #1,1 will be the default values
def f(x, coefficients=(1,1)):  
    a, b= coefficients
    return a+b*x

f(2,coefficients=(0,0))
f(2)


# =============================================================================
# SOME EXAMPLES FROM QUANT ECON
# =============================================================================

# EXERCISE 1
# compute inner product of x_vals, y_vals
x_vals = [2,4,3,5,0]
y_vals = [10,3,2,9,8]
inner_vals=[]
for x,y in zip(x_vals,y_vals):
     inner=x*y
     inner_vals.append(inner)
print (inner_vals)

#In one line count the even numbers in 0,99
sum (x%2==0 for x in range(100))

# given pairs = ((2,5),(4,2),(9,8),(12,10)) count number of pairs that have both numbers even.
pairs = ((2,5),(4,2),(9,8),(12,10))
sum(x%2==0 and y%2==0 for x,y in pairs)



# EXERCISE 2
# Function that computes a polynomial given coefficients and x value. Use enumerate()
def p(x,coeff):
    for i in enumerate(coeff):
        poly=coeff*x**(i-1)
        print (poly)
    return sum(poly)    #DOESN'T WORK

#Sargent's solution:
def p(x,coeff):
    return sum(a*x**i for i, a in enumerate(coeff))
coeff= [0 , 1]
p(1,coeff)



#EXERCISE 3
def f(string):
    count=0
    for letter in string:
        if letter==letter.upper() and letter.isalpha():
            count +=1
    return count
f('The Rain in Spain')

# EXERCISE 4
#Write function that takes 2 sequences and returns true if every element in seq_a i an element of seq_b
#do it without using sets and sets methods

seq_a= [1,2,3,4,5,6]
seq_b= [1,2,4,4,8,7,6,9]

def belongs(seq_a,seq_b):
    for a, in seq_a:
        if a==any(seq_b):
            return True
        else:
            return False

#Sargent's solution
def f(seq_a, seq_b):
    is_subset= True
    for a in seq_a:
          if a not in seq_b:
              is_subset= False
    return is_subset

print(f([1,2],[1,2,3]))
print(f([1,2,3],[1,2]))

#Using sets data type the solution is easier:
def f(seq_a,seq_b):
    return set(seq_a).issubset(set(seq_b))


#EXERCISE 5
#Piecewise interpolation
def linapprox(f, a, b, n, x):
    """
    Evaluates the piecewise linear interpolant of f at x on the interval
    [a, b], with n evenly spaced grid points.

    Parameters
    ===========
        f : function
            The function to approximate

        x, a, b : scalars (floats or integers)
            Evaluation point and endpoints, with a <= x <= b

        n : integer
            Number of grid points

    Returns
    =========
        A float. The interpolant evaluated at x

    """
    length_of_interval = b-a
    num_subintervals= n-1
    step = length_of_interval / num_subintervals
    
    # === find first grid pont larger than x===#
    point = a
    while point <=x:
        point += step
    # === x must lie between the gridpoints (point-step) and point === #
    u, v = point - step, point
    return f(u)+(x-u)*(f(v)-f(u))/(v-u)





