# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 16:39:50 2020

@author: rodri
"""



import numpy as np
import os
os.chdir('C:/Users/rodri/Dropbox/PhD thesis/python')
from data_functions_albert import data_stats
import quantecon as qe
import seaborn as sns
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import multiprocessing as mp
from scipy.interpolate import interp1d
from fixed_point import compute_fixed_point


#To create non-even spaced grids
def my_lin(lb, ub, steps, spacing=2):
    span = (ub-lb)
    dx = 1.0 / (steps-1)
    return np.array([lb + (i*dx)**spacing*span for i in range(steps)])


### Set-up:
folder = 'C:/Users/rodri/Dropbox/IDEA/lectures - TA/TA quant macro/TA6/'
save = False
plot = False
save_policies = True
save_simulation = False
plot_agg = False

### Accuracy and speed settings
num_cores = int(mp.cpu_count()-2)
tol_error = 0.04
T= 80
N= 100000

# Set seed for replicability
np.random.seed(1)



# Initialize model ====================================================
# Parameters
A = 160
alpha, rho, bbeta, delta, tau = 0.36, 1.5, 0.9, 0.02, 600
# alpha from Aiyagari (capital share US)

r, w = 0.005,  550
sigma = 0.7
c_bar = 10
z = 15.24


print(r<(1/bbeta-1))


#grids
N_x, N_l, N_a,  N_shocks = 50, 3, 500, 10000
b = 0.01
a_max = 80000
spacing=1


y = np.random.normal(0, sigma, N_shocks)
eps =  np.exp(y-sigma**2/2)

a_grid = my_lin(b+1e-2, a_max, N_a, spacing=spacing)

np.mean(eps)

x_min = 60
x_max = 80000
x_grid = my_lin(x_min, x_max, N_x, spacing)

location = np.linspace(0 , 1.5, N_l)




# Utility function -------------
def u_float(c):
        if (c-c_bar)<+1e-8:
            return -1e16
        else: 
            return (c-c_bar)**(1-rho)/(1-rho)
        


# delivers index of the element in array closest to value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
 

# Vectorization 
vec_u = np.vectorize(u_float)

# Set empty Value and Policy functions: ===
V_new = np.empty((N_x, N_l))
return_mkt = np.empty((N_a))
return_autk = np.empty((N_a))
market_yes = np.empty((N_x, N_l))
g_a = np.empty((N_x, N_l))
g_c = np.empty((N_x, N_l))

## Initial Guess
V_guess = np.zeros((N_x, N_l))
#V_guess = np.load(folder+'V_function.npy')



v_market = np.zeros((N_x, N_l))
v_aut = np.zeros((N_x, N_l))

a_grid_vect, eps_vect = np.meshgrid(a_grid, eps)

#% Standard VFI ---------------------------------

def bellman_operator(V, return_policies=False):
    
    for i_l in range(N_l):
        v_func = interp1d(x_grid, V[:,i_l], fill_value='extrapolate', kind='quadratic')
        for i_x in range(N_x):
            return_mkt[:] = vec_u(x_grid[i_x] -a_grid -tau*location[i_l]) +bbeta*np.mean(v_func((1+r-delta)*a_grid_vect +eps_vect*w), axis=0)                                                                                                 
            return_autk[:] = vec_u(x_grid[i_x] -a_grid) +bbeta*np.mean(v_func((1-delta)*a_grid_vect +eps_vect*A), axis=0)
        
            ### Household participates in the market
            if  np.nanmax(return_mkt) > np.nanmax(return_autk):
                                                     
                V_new[i_x, i_l] =  np.nanmax(return_mkt)
                market_yes[i_x,i_l] = 1
                g_a[i_x, i_l] = a_grid[np.argmax(return_mkt[:])]                  
                g_c[i_x, i_l] = x_grid[i_x] -tau*location[i_l] - g_a[i_x, i_l]
            
            ## Household stays in autarky
            else:
                V_new[i_x, i_l] = np.nanmax(return_autk)
                market_yes[i_x,i_l] = 0
                g_a[i_x, i_l] = a_grid[np.argmax(return_autk[:])]                  
                g_c[i_x, i_l] = x_grid[i_x] - g_a[i_x, i_l]
                
    if return_policies==True:
        return V_new, g_a, g_c, market_yes
    else:
        return V_new


#%%
qe.tic()
V = compute_fixed_point(bellman_operator, V_guess, max_iter=1000, error_tol=tol_error)
qe.toc()


qe.tic()
V_next, g_a, g_c, market_yes = bellman_operator(V, return_policies=True)
qe.toc()


def plot_policy_2d(grid, policy, policy_name, save=False, line_45=False, folder=folder):
        
    fig,ax = plt.subplots()
    for i_l in range(N_l):
        ax.plot(grid, policy[:,i_l], label = 'Location: '+str(location[i_l]))
    if line_45 == True:
        ax.plot(grid,grid, linestyle='dashed', label='45 line')
    ax.set_xlabel('Cash on hand')
    ax.legend()
    ax.set_title(policy_name)
    if save==True:
        fig.savefig(folder+policy_name+'.png')  
    plt.show() 


N_x_large = 20
X_GRID =  my_lin(x_min, x_max, N_x_large, spacing)
V_NEW = np.empty((N_x_large, N_l))
G_A = np.empty((N_x_large, N_l))
G_C = np.empty((N_x_large, N_l))
               
for i_l in range(0,N_l):
    v_func = interp1d(x_grid, V_next[:,i_l], fill_value="extrapolate")
    g_a_func  = interp1d(x_grid, g_a[:,i_l], fill_value="extrapolate")
    g_c_func = interp1d(x_grid, g_c[:,i_l], fill_value="extrapolate")
    
    V_NEW[:,i_l] = v_func(X_GRID)
    G_A[:,i_l]  = g_a_func(X_GRID)
    G_C[:,i_l]  = g_c_func(X_GRID)

if save_policies==True:
    np.save(folder+'g_a', g_a)
    np.save(folder+'g_c', g_c)
    np.save(folder+'V_function', V_next)
    np.save(folder+'V_FUNC', V_NEW)


save = False
plot_policy_2d(grid=x_grid, policy=V_next, save=save, policy_name='Value_function')
plot_policy_2d(grid=x_grid, policy=g_a, save=save, line_45=True, policy_name='Assets_Policy') 
plot_policy_2d(grid=x_grid, policy=g_c, save=save,  policy_name='Consumption_Policy')
plot_policy_2d(grid=x_grid, policy=market_yes, save=save,  policy_name='Accessing Market')




#%% Parallelization version



# Define class for the model
class modelState(object):
    def __init__(self, N_a, N_l, N_x, A, bbeta, w, r, delta, x_grid, a_grid, location,
                  eps, return_mkt, return_autk):
		 
       self.N_a = N_a
       self.N_x = N_x
       self.N_l = N_l
       self.eps =  eps
       self.A = A
       self.bbeta = bbeta
       self.delta = delta
       self.w = w
       self.r = r
       self.a_grid = a_grid
       self.location = location
       self.x_grid = x_grid
       self.return_mkt, self.return_autk = return_mkt, return_autk
       

# Given a indexed state of today it computes the optimal value function V(index).
def V_i(V, index):
    vec_u = np.vectorize(u_float)
    i_x, i_l = np.unravel_index(index, (N_x, N_l))
    
    a_grid_vect, eps_vect = np.meshgrid(a_grid, eps)
    v_func = interp1d(x_grid, V[:,i_l], fill_value='extrapolate', kind='quadratic')
    
    return_mkt[:] = vec_u(x_grid[i_x] -a_grid -tau*location[i_l]) +bbeta*np.mean(v_func((1+r-delta)*a_grid_vect +eps_vect*w), axis=0)                                                                                                 
    return_autk[:] = vec_u(x_grid[i_x] -a_grid) +bbeta*np.mean(v_func((1-delta)*a_grid_vect +eps_vect*A), axis=0)
        
    ### Household participates in the market
    if  np.nanmax(return_mkt) > np.nanmax(return_autk):
        return np.nanmax(return_mkt)
            
    ## Household stays in autarky
    else:
        return np.nanmax(return_autk)
     


### try with different backend and see fastest one:  backend='multiprocessing'          
def V_parallel(V):
        
        results = Parallel(n_jobs=num_cores, verbose=0)(delayed(V_i)(index=idx, V=V) for idx in range(0,N_x*N_l))
        V_new = np.array(results)
        V_new.shape = (N_x,N_l)

                        
        return V_new
    

states = modelState(N_a, N_l, N_x, A,  bbeta, w, r, delta, x_grid, a_grid, location,
                  eps, return_mkt, return_autk)


N_a, N_l, N_x = states.N_a, states.N_l, states.N_x
A,   bbeta, w, r, delta = states.A, states.bbeta, states.w, states.r, states.delta
x_grid, a_grid, eps = states.x_grid, states.a_grid, states.eps
return_mkt, return_autk = states.return_mkt, states.return_autk

### Increasing tolerance error didnt produce nicer policy functions: go again with a low tol_error
qe.tic()
V = compute_fixed_point(V_parallel, V_guess, max_iter=1000, error_tol=tol_error, save=True, save_name= 'v_itera5',folder=folder)
qe.toc()


qe.tic()
V_next, g_a, g_c, market_yes = bellman_operator(V, return_policies=True)
qe.toc()



def plot_policy_2d(grid, policy, policy_name, save=False, line_45=False, folder=folder):
        
    fig,ax = plt.subplots()
    for i_l in range(N_l):
        ax.plot(grid, policy[:,i_l], label = 'Location: '+str(location[i_l]))
    if line_45 == True:
        ax.plot(grid,grid, linestyle='dashed', label='45 line')
    ax.set_xlabel('Cash on hand')
    ax.legend()
    ax.set_title(policy_name)
    if save==True:
        fig.savefig(folder+policy_name+'.png')  
    plt.show() 


save_policies = False
if save_policies==True:
    np.save(folder+'g_a', g_a)
    np.save(folder+'g_c', g_c)
    np.save(folder+'V_function', V_next)
   


save = False
plot_policy_2d(grid=x_grid, policy=V_next, save=save, policy_name='Value_function')
plot_policy_2d(grid=x_grid, policy=g_a, save=save, line_45=True, policy_name='Assets_Policy') 
plot_policy_2d(grid=x_grid, policy=g_c, save=save,  policy_name='Consumption_Policy')
plot_policy_2d(grid=x_grid, policy=market_yes, save=save,  policy_name='Accessing Market')



#%%


N= 10000
T=500  #200
### Computes the stationary distribution of the economy by Montecarlo simulation
def INVARIANT_DISTR(grid, policy, T=200, N=10000):
    g_a = policy[0]
    g_c = policy[1]
    g_m = policy[2]
    

    eps_state = np.empty((T,N))
    a_state = np.empty((T,N))
    x_state = np.empty((T,N))
    y_state = np.empty((T,N))
    market_state = np.empty((T,N))
    hh = np.empty((T,N))
    time = np.empty((T,N))
    c_state = np.empty((T,N))
    mean_x = np.empty(T)
    var_x = np.empty(T)
    
        
    eps_state[0,:] = np.random.lognormal(1, sigma, N)
    market_state[0,:] = 0
    a_state[0,:] = a_grid[find_nearest(a_grid,1000)]
    
    x_state[0,:] = (1-delta)*a_state[0,:] +A*eps_state[0,:]
    loc = np.random.choice(range(0,N_l),size=N)

    for t in range(1, T):
        
        eps_state[t,:] = np.random.lognormal(1, sigma, N)
        
        for n in range(0,N):
            
            a_state[t,n] = g_a[find_nearest(grid,x_state[t-1,n]),loc[n]]
            c_state[t,n] = g_c[find_nearest(grid,x_state[t-1,n]),loc[n]] 
            market_state[t,n] = g_m[find_nearest(grid,x_state[t-1,n]),loc[n]]
            y_state[t,n] =  market_state[t-1,n]*(w*eps_state[t-1,n] -tau*location[loc[n]]) +(1-market_state[t-1,n])*(A*eps_state[t-1,n]) 
            x_state [t,n] = market_state[t-1,n]*((1+r-delta)*a_state[t-1,n]) +(1-market_state[t-1,n])*((1-delta)*a_state[t-1,n]) +y_state[t,n]
            hh[t,n] = n
            time[t,n] = t
                     
        mean_x[t] = np.mean(x_state[t,:])
        var_x[t]  = np.var(x_state[t,:])
       
        if t%10 == 0:
            print(t)
            
    return hh[T-10:T,:], time[T-10:T,:], loc, x_state[T-10:T,:],  y_state[T-10:T,:], a_state[T-10:T,:], c_state[T-10:T,:], market_state[T-10:T,:], mean_x, var_x



qe.tic()
hh_state, time_state, loc, x_state, y_state, a_state, c_state, market_state, mean_x, var_x = INVARIANT_DISTR(grid=x_grid, policy = [g_a, g_c, market_yes],T=T, N=N)
qe.toc()

periods = len(y_state)
loc_state =  np.tile(loc.transpose(), (periods,1))



### SAVE STATES RESULTS IN A DATAFRAME
data_simulation_dict = [('hh', (hh_state.transpose()).flatten()),
                        ('t', (time_state.transpose()).flatten()),
                        ('loc', (loc_state.transpose()).flatten()),
                        ('x', (x_state.transpose()).flatten()),
                        ('y', (y_state.transpose()).flatten()), 
                        ('a', (a_state.transpose()).flatten()),
                        ('c', (c_state.transpose()).flatten()),
                        ('market', (market_state.transpose()).flatten()),]

data_time = pd.DataFrame.from_dict(dict(data_simulation_dict))


if save_simulation == True:
    data_time.to_csv(folder+'states_stationary.csv')


data = data_time.loc[data_time['t']==T-1]



### Convergence to stationary invariant distribution =========

fig, ax = plt.subplots()
ax.plot(range(0,T), mean_x, label='cash on hand')
ax.legend()
ax.set_xlabel('Time')
# ax.set_ylim((-40,10))
ax.set_title('Average across time')   


fig, ax = plt.subplots()
ax.plot(range(0,T), var_x, label='cash on hand')
ax.legend()
ax.set_xlabel('Time')
# ax.set_ylim((-40,10))
ax.set_title('Variance across time') 

fig, ax = plt.subplots(figsize=(8,6))
for t in range(0,4):
    x_data = data_time.loc[data_time['t']==T-t,'x']   
    sns.distplot(np.log(x_data), label='Period '+str(T-t))
plt.title('Invariant distribution')
plt.xlabel('Cash-on-hand')
plt.ylabel("Density")
plt.legend()
plt.show() 


### Market clearing

K = np.sum(data.loc[data['market']==1,'a'])/N

L = np.sum(data['market'])/N

w_m = z*(1-alpha)*K**alpha*L**(-alpha)
r_m = z*alpha*K**(alpha-1)*L**(1-alpha)

print('Check market clearing ---------------------------')
print('Previous w,r where:'+str(w)+', '+str(r))
print('New ones are:')
print('w*='+str(w_m))
print('r*='+str(r_m))



### States distribution in stationarity  =============
def plot_distribution(state, state_name, save=False, folder=folder):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.distplot(state, label=state_name)
    plt.title('Distribution of'+state_name)
    plt.xlabel(state_name)
    plt.ylabel("Density")
    plt.legend()
    if save==True:
        fig.savefig(folder+'distribution'+state_name+'.png')
    return plt.show()


plot_distribution(data['y'], 'Output', save=save)  
plot_distribution(data['a'], 'Assets', save=save)  
plot_distribution(data['c'], 'Consumption', save=save)  
plot_distribution(data['market'], 'Market Participation', save=save)  


# Economy Outcomes Summary  ====================================


summary2 = data_stats(data[['y','a','c','market']])
print(summary2.to_latex())
