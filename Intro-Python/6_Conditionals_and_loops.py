# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:09:13 2019

@author: rodri
"""
import numpy as np


# =============================================================================
#  Conditional statements 
# =============================================================================

a = 10
b = 20
## simple statements
if a==b:
    print('a=b')
elif a>b:         
    print('a larger b')
elif a<b:
    print('a smaller than b')
else:            
    print('error')     
if a!=b:
    print('a not equal to b')
    
## Combined statements
## &: and ... both conditions must be satisfied  
if (a==10) and (a+b==30):
    print('a equals 10 and a plus b equal c')
else:
    print('not equal')
    
### |: or ... One of conditions must be satisfied   
if (a==10) or (b==10): 
    print('one equals 10')
else:
    print('none equal')
    
    
# =============================================================================
# Loops
# =============================================================================
## loops in python can be created with very different strategies. Let's do some examples:
N_x = 10
x_grid = np.linspace(20,30,N_x)

for x in x_grid:
    print(x) 
    
for i,x in enumerate(x_grid):
    print(i)
    print(x)
    
### Very useful to keep for VFI or euler in solving recursive models:
    '''
    Let V(a,y) be the value function at assets level a, and y and transitory productivity shock that follows a markov proc.
    Let w be the wage of the individual.
    Then to compute the value function in discrete method we would:
        for i_a, a in enumerate(a_state):    ##loop over state space
            for i_y, y in enumerate(y_state):
                for i_ga, g_a in enumerate(a_state):  ###loop over possible choices assets tomorrow (a')
                    return_matrix[i_a, i_y, i_ga] = u(w*y +(1+r)*a -g_a) +beta*(M[i_y,:]@V[i_ga,:])    ## compute return matrix
                V[i_a,i_y] = np.nanmax(return_matrix[i_a,i_y,:])   #compute value function tomorrow: the one with the asset choice g_a 
                                                                   # that maximizes the value function.
        
    enumerate might be a little bit slower than range. For big problems where speed is very important, better use range.
    '''
    
    
## range function is also very useful for loops:
for x in range(0,N_x):
    print(x)
# equivalently:
for x in range(0,len(x_grid)):
    print(x)      
for i, x in enumerate(range(20,30)):
    print(i)
    print(x)  
 
    
## While loop:
count = 0
while count < 10:
    count += 1
    print(count)  


# we can also loop over lists:
list_sentence = ['Bananas', 'Potatoes', 'Maize']
for item in list_sentence:
    print('I like '+item)


## to store results from a loop, might be useful to create and empty list and then append. 
#(or create an empty array but we need to know dimension of results)
grid_100 = np.linspace(1,100,100)
list_even = []
for x in grid_100:
    if x%2==0:   # decorator % delivers the residual from a division
        list_even.append(x)


## Zip function: 
#The zip() function returns a zip object, which is an iterator of tuples where the first item
# in each passed iterator is paired together, and then the second item in each passed iterator are paired together etc.
#If the passed iterators have different lengths, the iterator with the least items decides the length of the new iterator.
        
x_grid = [1,2,3]
y_grid = [4,5,6,7]

for x in x_grid:
    for y in y_grid:
        result = x*y
        print(str(result))
        
for x, y in zip(x_grid,y_grid):
        result = x*y
        print(str(result))        
        
## to do reversed loops. We will have to to solve life-cycle models:
for x in reversed(range(0,N_x)):
    print(x) 
    



   
    
    