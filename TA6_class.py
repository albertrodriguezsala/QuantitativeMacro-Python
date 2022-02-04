# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:09:35 2018

@author: rodri
"""


import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe


class eatingup_urban_class:
    '''
    #----- GRID SIZES ----- #
    nperiod: Number of periods
    N_a: Number of grid points for assets
    N_z: Number of grid points for Z
    N_w: Number of grid points for Initial Productivity
    N_sim: Number of simulated samples    
      
    
    #----- UTILITY PARAMETERS----- #      
    β: Discount factor (0.95 anually)
    η_1, η_2: CRRA coefficients  
    κ: Utility weight on manufacturing goods.      
    p_f: Price of agricultural good.

    
    #----- SHOCK PARAMETERS -----#
           

    OUTPUT ------------------------------------------------------------------------------------------
    Initialize the class for the household rural problem.
    
    __init__ : Initialize the parameters and the global variables to solve the household rural problem.
    
    Basic functions: Imports the functions to use.
    
    plot_policies: Function to plot policies.
        
    solve_rural_lifecycle_problem: Solver of the rural household lifecycle problem.
    '''
    
    def __init__(self, nperiod=8,N_a=100, N_z=3, N_w=3, N_sim=10000,             
                 β=0.77, η_1=3, η_2=5, κ=0.1, a_bar=0, a_max= 10000,
                 pf=1, rrate = 1/0.77 -1,
                mean_logy_u=6.91, var_logy_u=0.9, mean_logk_u=4.51, var_logk_u=3.47, gy_u = np.array([0, 0.23, 0.31, 0.1, 0.01, 0.24, 0.05,-0.11]),  
                ρ =0.95, sig_z = 0.4):
    
        
        self.nperiod, self.N_a, self.N_z, self.N_w, self.N_sim = nperiod , N_a, N_z, N_w, N_sim
        self.β, self.η_1, self.η_2, self.κ, self.a_bar = β, η_1, η_2, κ, a_bar
        self.pf, self.rrate = pf, rrate
        self.cum_gy_u = np.cumsum(gy_u)
        c_bar= 0.001*np.exp(mean_logy_u)
        self.c_bar = c_bar
        
        
        #----- INITIAL DISTRIBUTION -----#
        mean_lnw_u      = mean_logy_u   
        self.mean_lnw_u = mean_lnw_u 
        var_lnw_u      = var_logy_u
        self.var_lnw_u = var_lnw_u 
        mean_lna0_u     = mean_logk_u
        self.mean_lna0_u = mean_lna0_u 
        var_lna0_u      = var_logk_u
        self.var_lna0_u  = var_lna0_u 
        
        
        #---- EMPTY POLICY FUNCTION ---- #
        self.empty_policy = 1e-8*np.ones((N_a,N_z,N_w,nperiod))
        
        #----- CONSTRUCT STATE SPACES: A, Z, W -----#
        a_min           = a_bar
        self.k_min      = a_min
        #a_max           = np.exp(mean_lna0_u + 3.6*np.sqrt(var_lna0_u))
        self.a_max      = a_max
        a_lin           = np.linspace(np.log(a_min - a_min + 1),np.log(a_max - a_min + 1),N_a)
        a_state         = np.exp(a_lin) - 1 + a_min
        self.a_state    = a_state
        
        mc              = qe.tauchen(rho=ρ,sigma_u=sig_z,n=N_z)
        self.mc         = mc
        self.sig_z      = sig_z
        self.ρ          = ρ
        z_state         = mc.state_values
        self.z_state    = z_state
        self.z_prob     = mc.P
        
        w_min           = np.maximum(mean_lnw_u-4.3*np.sqrt(var_lnw_u),np.log(c_bar+1e-5)-z_state[0])
        self.w_min      = w_min
        w_max           = mean_lnw_u + 4*np.sqrt(var_lnw_u)
        w_state         = np.linspace(w_min,w_max,N_w)
        self.w_state    = w_state
        self.Z,self.A,self.W = np.meshgrid(z_state,a_state,w_state,indexing='xy')



    def egm_urban_func(self,param,a0,y0,a1):
        
        cf      = param
        # COMPUTE cm FROM INTRATEMPORAL OPTIMAL CONDITION
        cm      = (((cf-self.c_bar)**(-self.η_1))/(self.κ*self.pf))**(-1/self.η_2)
        #INCOME SIDE OF BUDGET CONSTRAINT
        RHS1    = y0 + (1+self.rrate)*a0
        # EXPENDITURE SIDE OF BUDGET CONSTRAINT
        LHS1    = self.pf*cf + cm + a1
        # BUDGET CONSTRAINT MUST HOLD IN EQUALITY
        error   = abs(LHS1 - RHS1)
        return error
    
    def plot_policies(self, policy, policy_name, it, fix_w =1, folder='C:/Users/rodri/Dropbox/Eating UP Productivity/Model/python_Albert/figures/urban/', line_45 = False, fixi=0, legend=False, save=False):
         """
        INPUTS:
            policy[array]: 
                Policy, value function or other array to plot. must be shape [n_k,n_z,n_w,n_t]          
            policy_name [string]: 
                Name of the array to appear in the title, legend and y_axis of the figure.  
            it [int]: 
                Period of time to plot  
            fix_w [1 or anyting]: 
                To have w_state as fix (default as lowest state). If w is not fixed, then state z is fixed.      
            fixi [int]:
                Index of the state to fix.             
            Legend [Boolean]:
                Plot legend or not.
            
        OUTPUTS:
            Figure of the plot of the policy function.
        """
         if fix_w==1:
            fig, ax = plt.subplots(figsize=(10,7))
            for iz in range(0,self.N_z):             
                ax.plot(self.a_state,policy[:,iz,fixi,it], label='z='+str(round(self.z_state[iz],ndigits=2)))               
                ax.set_xlabel('Assets State')
                ax.set_ylabel(policy_name)
                ax.set_title(policy_name+' at period '+str(it+1)+' w ='+str(round(self.w_state[fixi],ndigits=2)))               
                if legend==True:
                    ax.legend()
            if line_45==True:
                ax.plot(self.a_state,self.a_state, linestyle='dashed', label='45 line')
            if save == True:
                fig.savefig(folder+policy_name+str(it+1)+'w='+str(round(self.w_state[fixi],ndigits=2))+'.png')    
            return plt.show() 
        
         else:
            fig, ax = plt.subplots()
            for iw in range(0,self.W_z):             
                ax.plot(self.a_state,policy[:,fixi,iw,it], label='w='+str(round(self.w_state[iw],ndigits=2)))
                ax.set_xlabel('Assets State')
                ax.set_ylabel(policy_name)
                ax.set_title(policy_name+' at period '+str(it+1)+' z = '+str(round(self.z_state[fixi],ndigits=2)))
                if legend==True:
                    ax.legend()
            if save == True:
                fig.savefig(folder+policy_name+str(it+1)+' fixed_z_urban.png', bbox_inches='tight')
            return plt.show()
    
    
    
    
    def plot_sim(self,data, name, folder='C:/Users/rodri/Dropbox/Eating UP Productivity/Model/python_Albert/figures/urban/', save=False ):
        fig, ax = plt.subplots()    
        ax.plot(np.array(range(0,self.nperiod)), data, label='Model')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_title(' Simulation of the Model: Lifecycle'+name)
        if save == True:
            fig.savefig(folder+'Simulation'+name+'urban.png')
            return plt.show()
            