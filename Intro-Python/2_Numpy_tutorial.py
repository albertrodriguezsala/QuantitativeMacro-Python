# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 14:28:30 2017

@author: Albert
"""
import numpy as np

# =============================================================================
#  Notes on Numpy
# =============================================================================

 # The basic unit of work in Numpy are arrays. Numpy arrays power a large proportion of the scientific python environment.
 # Numpy arrays are similar to lists but: The data must be of the same type and one of the types provided by dtype (types on numpy).
 #The main dtypes are: float, integers, boolean.

 # functions in numpy act element-wise, they are called vectorized functions. Also called ufuncs in numpy-speak.
x= 1

X= np.array([[1,2,3],[4,5,6]])
np.log(x)
np.log(X)

#Some examples --------------------

 #Zero matrix
a= np.zeros(10)

np.exp(a[0:3])
type(a[0])
 #to specify type:
a= np.zeros(10, dtype=int)
type(a[0])
 #to know dimension
a.shape
 #And to change it: The number of elements must be the same: if it was 10,1 now we can do 5,2 or 2,5
a.shape = (5,2)

#Ones matrix
b=np.ones((2,2))

#NaN matrix
c=np.empty(9)  #The numbers appeared are garbage numbers
c.shape=(3,3)

c = np.array([[],[]])

#identity matix
I=np.eye(4)
I=np.identity(3)

#linspace:
a_grid = np.linspace(0,100000,100)  ## to construct grids

#To create arrays from Python lists, tuples:
a= np.array([[1,2,3],[4,5,6]])  #2x3 array (or Matrix)
a= np.array((1,2,3), dtype=float)


#Indexing is the same as lists(starting on 0, negative, etc) and of similar sintaxi to matlab indexing:
a[:,0]

#We can use an array of dtype bool to extract elements from arrays/matrices:
#Example: Extract 1st column as before:
sel=np.array([[1,0,0],[1,0,0]], dtype=bool)
sel=np.array([[True, False, False],[True, False, False]])

col1=a[sel]

#to set all numbers of an array equal to x:
c[:]=29
I[:]=32

## Matrix multiplication in python
A= [[1,2,3],[3, 3, 9]]
A=np.matrix(A)
B= [[1,2],[2, 2],[1,1,]]
B= np.matrix(B)
np.dot(A,B)
# @ also works for multiplication of matrices. 
A @ B

# We can also use it to do the inner product of 2 flat arrays.
C = np.array((1,2))
D = np.array((10,20))
C@D

x=np.random.uniform(0,1, size=1000000)
x.mean()


#1.2 Array Methods ---------------------------------
#Some useful methods for arrays. Recall that method on an object/array are called by object.method:
A = np.array((4, 3, 2, 1))
A.sort() #Changes the order of the array
A.sum()  
A.mean()  
A.max()  
A.argmax()   ## location of the maximum value. Remember we start in 0.. so 4th position is the 3th.
A.cumsum()
A.cumprod() 
A.var()
A.std()  
A.shape=(2,2)
A.T #Equivalent to A.transpose

#Another important method is searchsorted()
z = np.linspace(2,4,5)
z.searchsorted(3) #returns the index of the first ellement of z that is >=3


#We also have commands to copy elements.
a=np.random.randn(3)
b=np.empty_like(a)  #empty array of same size as a
np.copyto(b,a)  #Now b is an independent copy (also called a deep copy)

# Alternatively
b = np.copy(a)

### IN PYTHON IDENTITIES WORK BOTH DIRECTIONS: A = B: change B modifies A. but also change A modifies B!
#independent copies are useful because now we can modify b without changing a:  
b[:]=2
print(a, b)

### VERY IMPORTANT:
### FOR EXAMPLE AT ITERATING VALUE FUNCTION... IF WE SET V(j+1) = V(j), python identifies these 2 objects as equal
# Therefore at modifying one we modify both... We need to set it up as: V(j+1) = np.copy(V(j))


## Numpy functions ----------------------------------------------
#numpy provides version of standard functions that are element-wise
z = np.array ([1,2,3])
np.sin(z)
np.log(z)
np.exp(z)

#Since these functions act element-wise, they are called vectorized functions. Also called ufuncs in numpy-speak.
 
#However, not all user defined functions act element-wise:
def f(x):
    return 1 if x>0 else 0

a=np.array((1,2,-2))
f(a)  #we obtain error.

#numpy function np.where provides a vectorized alternative:
x = np.random.randn(4)
np.where(x>0,1,0)   #Insert 1 if x>0 true, othw 0.


#We can also vectorize a function with np.vectorize: VERY USEFUL TO SPEED-UP OUR CODE
def f(x): return 1 if x>0 else 0
f=np.vectorize(f)  #However this system van be more slow than others.
f(a)

## If time allows check example utility function.

# Comparisons
#in general comparisons on arrays are done element-wise
# also for < > <= >=
z=np.array([2,3])
y=np.array([2,3])
z==y
y[0]=5  
z==y
z!=y

#We can also do comparisons against scalars:
z=np.linspace(0,10,5)
b= z>3

#Many useful commands in numpy are stored in subpackages:
np.linalg.det(A)

















