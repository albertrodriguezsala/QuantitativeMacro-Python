# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:38:51 2019

@author: rodri
"""
# =============================================================================
# Solving Neoclassical Model of Growth in Recursive Formulation
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
from scipy.optimize import minimize, minimize_scalar
from scipy.interpolate import interp1d

# Parameters
theta = 0.679
beta = 0.988
delta = 0.013
ka = 5.24
vue = 2

# Functions
def y_func(k, h):
    return k**(1-theta)*h**theta


def u(c, h):
    if c<1e-2:
        return -1e16      # I needed to impose this condition. if not I was getting c<0 so V(k)=nan
    else:
        return  np.log(c) -ka*((h**(1+(1/vue)))/(1+(1/vue))) 


## Iteration settings
tol_error = 0.001
itera_max = 1000




#%% Discrete VFI with inelastic labor ==============================================

h=1

# Grids
N_k = 50
N_h = 50
k_grid = np.linspace(0.01, 10, N_k)

# Initial value functions 
V_guess = np.zeros((N_k))
 
### Empty policy functions
k_policy = np.empty((N_k))
h_policy = np.empty((N_k))
c_policy = np.empty((N_k))
V_next = np.zeros((N_k))

### Empty Return matrix
m = np.zeros((N_k,N_k))  


#Bellman contractor
def bellman_operator(V,return_policies=False):
    
    
    for ik, k in enumerate(k_grid):     #iterate on state
        for igk, gk in enumerate(k_grid):   #iterate on choice
            ### Fill-up return matrix
            m[ik,igk] =  u((y_func(k,h) +(1-delta)*k - gk),h) +beta*V[igk]
            
        V_next[ik] = np.nanmax(m[ik,:])        
        k_policy[ik] =  k_grid[np.unravel_index(np.argmax(m[ik,:], axis=None), m[ik,:].shape)[0]]   # get the argmax from 2nd dimension
        c_policy[ik] =  y_func(k,h) +(1-delta)*k - k_policy[ik]  
        
    if return_policies==True:
        return V_next, k_policy, h_policy, c_policy
    else:
        return V_next

## Compute fixed point in the bellman equation
qe.tic()
V = qe.compute_fixed_point(bellman_operator, V_guess, max_iter=itera_max, error_tol=tol_error, print_skip=20)
V_next, g_k, g_h, g_c = bellman_operator(V, return_policies=True)
qe.toc()



### Plots
fig, ax = plt.subplots()
ax.plot(k_grid,V, label='V(k)')
plt.show()

fig, ax = plt.subplots()
ax.plot(k_grid,g_c, label='c(k)')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(k_grid,g_k, label='k1(k)')
ax.legend()
plt.show()



#%% Discrete VFI with elastic labor ==============================================

# Grids
N_k = 50
N_h = 50
k_grid = np.linspace(0.01, 10, N_k)
h_grid = np.linspace(0.01, 1, N_h)


# Initial value functions 
V_guess = np.zeros((N_k))
 

### Empty policy functions
k_policy = np.zeros((N_k))
h_policy = np.zeros((N_k))
c_policy = np.zeros((N_k))
V_next = np.zeros((N_k))


### Empty Return matrix
m = np.zeros((N_k,N_k,N_h))   # extra-dimension from labor choic

    
## To increase speed, Vectorize (ie no need to loop for all choice variables)
u_vect = np.vectorize(u)   #Given that my utility function has a conditional statement, python cannot vectorized by default. I need to call np.vectorize() to do so.

#Bellman equation
def bellman_operator(V,return_policies=False):


    for ik, k in enumerate(k_grid):  #loop on state
        for igh, gh in enumerate(h_grid):   #loop only one choice variable, h. k' choice is vectorized.
            ### Fill up return matrix
            m[ik,:,igh] =  u_vect((y_func(k,gh) +(1-delta)*k - k_grid),gh) +beta*V
            
        V_next[ik] = np.nanmax(m[ik,:,:])        
        k_policy[ik] = k_grid[np.unravel_index(np.argmax(m[ik,:,:], axis=None), m[ik,:,:].shape)[0]]  # argmax of k' dimension
        h_policy[ik] = h_grid[np.unravel_index(np.nanargmax(m[ik,:,:], axis=None), m[ik,:,:].shape)[1]]  # argmax of h dimension
        c_policy[ik] =  y_func(k,h_policy[ik]) +(1-delta)*k - k_policy[ik]          
    if return_policies==True:
        return V_next, k_policy, h_policy, c_policy
    else:
        return V_next

## Compute fixed point in the bellman equation
qe.tic()
V = qe.compute_fixed_point(bellman_operator, V_guess, max_iter=itera_max, error_tol=tol_error, print_skip=20)
V_next, g_k, g_h, g_c = bellman_operator(V, return_policies=True)
qe.toc()



### Plots
fig, ax = plt.subplots()
ax.plot(k_grid,V, label='V(k)')
plt.show()

fig, ax = plt.subplots()
ax.plot(k_grid,g_h, label='h(k)')
ax.plot(k_grid,g_c, label='c(k)')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(k_grid,g_k, label='k1(k)')
ax.legend()
plt.show()


#%% Continuous VFI solution (inelastic labor) ==============================================

h=1

#Working with continuous methods we work with smaller grids
# Grids
N_k = 20
k_grid = np.linspace(0.01, 10, N_k)


V_guess = np.zeros((N_k))

### Empty policy functions
V_next = np.zeros((N_k))
k_policy = np.zeros((N_k))
c_policy = np.zeros((N_k))


#Bellman equation
def bellman_operator_cont(V_array,return_policies=False):
      
    v_func = lambda x: np.interp(x, k_grid, V_array)
    #v_func = interp1d(k_grid, V_array, fill_value='extrapolate', kind='quadratic')
 
    for ik, k in enumerate(k_grid):
        
        def criterion_func(c):  ## on c
            k_tomorrow = y_func(k,h) +(1-delta)*k -c   
            return  -(u((c),h) +beta*v_func(k_tomorrow))
        
        c_max = y_func(k,h) +(1-delta)*k
        result = minimize_scalar(criterion_func, bounds=(1e-2, c_max), method='bounded')
        
        c_policy[ik] = result.x
        V_next[ik] =  -result.fun   
        k_policy[ik] =  y_func(k,h) +(1-delta)*k - c_policy[ik]  
        
    if return_policies==True:
        return V_next, k_policy, c_policy
    else:
        return V_next



## Compute fixed point in the bellman equation
qe.tic()
V = qe.compute_fixed_point(bellman_operator_cont, V_guess, max_iter=itera_max, error_tol=tol_error, print_skip=20)
V_2, g_k, g_c = bellman_operator_cont(V, return_policies=True)
qe.toc()

N_k_large = 100
K_GRID = np.linspace(0.01, 10, N_k_large)


V_star = np.interp(K_GRID,k_grid, V_2)
g_k =  np.interp(K_GRID,k_grid, g_k)
g_c =   np.interp(K_GRID,k_grid, g_c)



### Plots
fig, ax = plt.subplots()
ax.plot(K_GRID,V_star, label='V(k)')
plt.show()

fig, ax = plt.subplots()
ax.plot(K_GRID,g_c, label='c(k)')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(K_GRID,g_k, label='k1(k)')
ax.legend()
plt.show()






#%% Continuous VFI solution: (elastic labor) ==============================================


def u(c, h):
    return  np.log(c) -ka*((h**(1+(1/vue)))/(1+(1/vue))) 


#with continuous methods we work with smaller grids

# Grids
N_k = 20
k_grid = np.linspace(0.01, 10, N_k)
V_guess = np.zeros((N_k))

### Empty policy functions
V_next = np.zeros((N_k))
k_policy = np.zeros((N_k))
c_policy = np.zeros((N_k))
h_policy = np.zeros((N_k))

#Bellman equation
def bellman_operator_cont(V_array,return_policies=False):
      
    v_func = lambda x: np.interp(x, k_grid, V_array)
 
    for ik, k in enumerate(k_grid):
 
        def criterion_func(x):  ## on k' 
            c = x[0]
            h = x[1]
            k_tomorrow = y_func(k,h) +(1-delta)*k -c
            return  -(u((c),h) +beta*v_func(k_tomorrow))
        
        c_max = y_func(k,1)+(1-delta)*k
        bnds = ((1e-2, c_max), (1e-2, 1))
        x0 = [y_func(k,1), 1]
        result = minimize(criterion_func, x0=x0, bounds=bnds, method='L-BFGS-B')
        
        c_policy[ik] = result.x[0]
        V_next[ik]   =  -result.fun 
        h_policy[ik] = result.x[1]
        k_policy[ik] =  y_func(k,h_policy[ik]) +(1-delta)*k - c_policy[ik]  
        
    if return_policies==True:
        return V_next, k_policy, c_policy, h_policy
    else:
        return V_next




## Compute fixed point in the bellman equation
qe.tic()
V = qe.compute_fixed_point(bellman_operator_cont, V_guess, max_iter=itera_max, error_tol=tol_error, print_skip=20)
V_2, g_k, g_c, g_h = bellman_operator_cont(V, return_policies=True)
qe.toc()

N_k_large = 100
K_GRID = np.linspace(0.01, 10, N_k_large)

# Note that we solved for the nodes. Now we need to interpolate

V_star = np.interp(K_GRID,k_grid, V_2)
g_k =  np.interp(K_GRID,k_grid, g_k)
g_c =   np.interp(K_GRID,k_grid, g_c)
g_h =   np.interp(K_GRID,k_grid, g_h)

### Plots
fig, ax = plt.subplots()
ax.plot(K_GRID,V_star, label='V(k)')
plt.show()

fig, ax = plt.subplots()
ax.plot(K_GRID,g_c, label='c(k)')
ax.plot(K_GRID,g_h, label='h(k)')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(K_GRID,g_k, label='k1(k)')
ax.legend()
plt.show()











#%% EXTRA: Policy iteration in continuous. Also called FOC iteration, time iteration, etc.



#Working with continuous methods we work with smaller grids
# Grids
N_k = 20
N_h = 20
k_grid = np.linspace(1, 10, N_k)

### Inelastic labor supply 
h=1


def u(c, h):
    return  np.log(c) -ka*((h**(1+(1/vue)))/(1+(1/vue))) 

def euler_iterator(g_c):
    c_policy = np.empty((N_k))    
    g_func = lambda x: np.interp(x, k_grid, g_c)
        
    for ik, k in enumerate(k_grid):
        def criterion_func(c):  ## on consumption today 
            k_tomorrow =  y_func(k,h) +(1-delta)*k -c  ##function of c today and labor tomorrow
            RHS = (g_func(k_tomorrow))*(beta*(1-theta)*k**(-theta)*h**(theta)+1-delta)**(-1)
            
            return (c - RHS)**2
        x0 = [y_func(k,h)]
        c_policy[ik] = minimize(criterion_func, x0=x0).x
        #c_max = y_func(k,h) +(1-delta)*k
        #c_policy[ik] = minimize_scalar(criterion_func, bounds=(1e-2, c_max), method='bounded').x
     
        
        
    return c_policy



itera=0
tol_c = 1e-2
g_c_list = []
g_h_list = []

g_c_guess = k_grid
g_c_list.append(g_c_guess)


while itera <= 10000: 
    print('iter'+str(itera))
    
    g_c = euler_iterator(g_c_list[itera])
    deviation_c = max(np.abs(g_c - g_c_list[itera])) 
    
    print('deviation c(k):'+str(deviation_c))
    
    if (deviation_c<=tol_c).all():
        print('Convergence reached')
        break
    
    g_c_list.append(g_c)
    itera = itera + 1
 
    
K_grid =  np.linspace(0.01, 10, 50)      
g_c = np.interp(K_grid, k_grid, g_c_list[-1])         
 
g_k = K_grid**(1-theta)*h**theta +(1-delta)*K_grid -g_c
V = u(g_c, h)   

### Plots
fig, ax = plt.subplots()
ax.plot(K_grid,V, label='V(k)')
plt.show()

fig, ax = plt.subplots()
ax.plot(K_grid,g_c, label='c(k)')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(K_grid,g_k, label='k1(k)')
ax.legend()
plt.show()
            




        


