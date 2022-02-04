# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 12:46:30 2017

@author: Albert
"""
# PROBLEM SET 2 #############


#Clear
#reset

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

#Parameters
theta = 0.67
h= 0.31
beta = 0.98
z = 1

#Define grid
grid_max = 10
grid_points = 200
grid_k = np.linspace(1e-5, grid_max, grid_points) #we start at 0+eps


#Utility function   
def u_prime(c):
     return 1/c

#Production function
def y(k,z):
     return k**(1-theta)*(z*h)**theta


## NOTE: As learnt from students, z dissapears from these equations. Thus anyvalue works.
     ## To calibrate values normalize y=1 and also use beta as calibration parameter.

# Solve for the SS and calibrate productivity z and depreciation delta
def k_star(z):
    ### where z[1] = delta,  z[0]=z
    return ((1/beta-1+z[1])/((1-theta)*(z[0]*h)**theta))**(-1/theta)

def SS(z):
    ky_ratio =  4  - (k_star(z)/z[0]*h)**theta
    iy_ratio = 0.25-  (z[1]*k_star(z)/(z[0]*h))**theta
    
    return  np.array([ky_ratio, iy_ratio])

x0=[2, 0.5]    

### Use scipy.fsolve to find the values of z and delta that make ky_ratio, iy_ratio equations to 0
param = fsolve(func=SS,x0=x0) # z and delta

K_SS=k_star(param)

delta=param[1]
z1=param[0]
z2=2*z1
K_SS2=k_star([z2, delta])



#%% =============================================================================
#  c. Transition after a productive shock
# =============================================================================


n=100 #periods for transition


def transition_law_c(x, n=n):
    z=z2
    k_0 = K_SS
    k_end = K_SS2
    x[0] = K_SS
    x[n-1] = K_SS2
    k_vector = np.empty(n)
    for i in range(0,n-2):
    #First period
        if i==0:
            k_vector[i+1]=u_prime(y(k_0,z)+(1-delta)*k_0-x[i+1])-beta*u_prime(y(x[i+1],z)+(1-delta)*x[i+1]-x[i+2])*(1-delta+(1-theta)*x[i+1]**(-theta)*(z*h)**theta)    
    #last
        elif i==(n-2):
            k_vector[i+1]=u_prime(y(x[i],z)+(1-delta)*x[i]-x[i+1])-beta*u_prime(y(x[i+1],z)+(1-delta)*x[i+1]-k_end)*(1-delta+(1-theta)*x[i+1]**(-theta)*(z*h)**theta)
    #rest of periods:
        else:
            k_vector[i+1]=u_prime(y(x[i],z)+(1-delta)*x[i] -x[i+1])-beta*u_prime(y(x[i+1],z)+(1-delta)*x[i+1]-x[i+2])*(1-delta+(1-theta)*x[i+1]**(-theta)*(z*h)**theta)

    return k_vector
 

x0 = np.ones(n)*K_SS
z_array = np.ones(n)*z2
k_array  =  fsolve(transition_law_c,x0)


y_array= y(k_array, z_array)
i_array = np.zeros(n)
i_array[0:n-1] =k_array[1:n]-(1-delta)*k_array[0:n-1]
c_array = y_array[0:n] - i_array



fig,ax = plt.subplots()
ax.plot(np.array(range(1,n+1)), k_array)
plt.show()

fig,ax = plt.subplots()
ax.plot(np.array(range(1,n+1)), y_array, label='y')
ax.plot(np.array(range(1,n+1)), i_array, label='i')
ax.plot(np.array(range(1,n+1)), c_array, label='c')
ax.legend()
plt.show()


#%% =============================================================================
# d. Transitions with an unexpected shock 
# =============================================================================
n=25
shock_t = 10

def transition_law_d(x, n=n):
    z=z2
    k_0 = K_SS
    k_end = K_SS
    x[0] = K_SS
    x[n-1] = K_SS
    k_vector = np.empty(n)
    for i in range(0,n-2):
    #First period
        if i==0:
            k_vector[i+1]=u_prime(y(k_0,z)+(1-delta)*k_0-x[i+1])-beta*u_prime(y(x[i+1],z)+(1-delta)*x[i+1]-x[i+2])*(1-delta+(1-theta)*x[i+1]**(-theta)*(z*h)**theta)    
    #last
        elif i==(n-2):
            k_vector[i+1]=u_prime(y(x[i],z)+(1-delta)*x[i]-x[i+1])-beta*u_prime(y(x[i+1],z)+(1-delta)*x[i+1]-k_end)*(1-delta+(1-theta)*x[i+1]**(-theta)*(z*h)**theta)
    #rest of periods:
        else:
            if i>=shock_t:
                z=np.copy(z1)
            k_vector[i+1]=u_prime(y(x[i],z)+(1-delta)*x[i] -x[i+1])-beta*u_prime(y(x[i+1],z)+(1-delta)*x[i+1]-x[i+2])*(1-delta+(1-theta)*x[i+1]**(-theta)*(z*h)**theta)

    return k_vector
 

x0 = np.ones(n)*K_SS
z_array = np.ones(n)*z2
z_array[shock_t:n] = z1
k_array  =  fsolve(transition_law_d,x0)

#

y_array= y(k_array, z_array)
i_array = np.zeros(n)
i_array[0:n-1] =k_array[1:n]-(1-delta)*k_array[0:n-1]
c_array = y_array[0:n] - i_array





fig,ax = plt.subplots()
ax.plot(np.array(range(1,n+1)), k_array)
plt.show()

fig,ax = plt.subplots()
ax.plot(np.array(range(1,n+1)), y_array, label='y')
ax.plot(np.array(range(1,n+1)), i_array, label='i')
ax.plot(np.array(range(1,n+1)), c_array, label='c')
ax.legend()
plt.show()












