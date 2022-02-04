# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:39:02 2019

@author: rodri
"""
# =============================================================================
# Taylor Approximation of functions
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.misc import derivative 

N_x = 50
x_grid = np.linspace(0.01,2,N_x)
x_bar = 1

def f_x(x):
    return x**0.321


### Taylor approximation
def taylor_approx(fun, x_grid, x_bar, degree):    
    f_approx = np.empty(N_x)
    ### compute series of derivatives:  
    for i,x in enumerate(x_grid):
        h = x-x_bar
        f_approx[i] = x_bar
        for n in range(1,degree+1):   
            f_approx[i] += derivative(func=fun, x0=x_bar, n=n, order=15)*(h)**n/math.factorial(n)  
    return f_approx

y_real = f_x(x_grid)


## Compute Taylor approximation for different degrees of aproximation
degrees = range(1,5+1)
taylor_approx_list = []
fx= np.empty((N_x))
for degree in degrees:
    fx = taylor_approx(f_x,x_grid,x_bar,degree)
    taylor_approx_list.append(fx)
    
 
### Plot approximations vs real function
fig, ax = plt.subplots()
ax.plot(x_grid,y_real, label = 'F(x)')
for i,fx in enumerate(taylor_approx_list):
    ax.plot(x_grid,fx, linestyle='dashed', label = 'Taylor Approx'+str(i+1))
    ax.legend()
    ax.set_ylim((-1,2))
plt.show()

