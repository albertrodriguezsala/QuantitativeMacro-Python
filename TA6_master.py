""" 
   -*- coding  utf-8 -*-

Created on Fri Dec 14 17 04 40 2018

@author  rodri
"""
# =============================================================================
#    Solve the Urban Household Problem
# =============================================================================
import os
os.chdir('C:/Users/rodri/Dropbox/Eating UP Productivity/Model/python_Albert')
from class_urban import eatingup_urban_class
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
import quantecon as qe
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import Rbf

my_interpolation = LinearNDInterpolator

folder='C:/Users/rodri/Dropbox/Eating UP Productivity/Model/python_Albert/figures urban/'
# 
# ---- Initialize urban Households Problem Class: ---- #

cp = eatingup_urban_class(N_a=50, N_z=3, N_w=3)

# GRIDS
nperiod,N_a, N_z, N_w = cp.nperiod, cp.N_a, cp.N_z, cp.N_w

# UTILITY PARAMETERS
β, c_bar = cp.β,  cp.c_bar
η_1, η_2, κ =  cp.η_1, cp.η_2, cp.κ

# STATES
A, Z, W, k_state, z_state, z_prob, w_state = cp.A, cp.Z, cp.W, cp.a_state, cp.z_state, cp.z_prob, cp.w_state
cum_gy_u = cp.cum_gy_u

#----- VALUE FUNCTION, ENVELOPE CONDITION, AND POLICY FUNCTIONS -----#
V_urban         = 1e-8*np.ones((N_a,N_z,N_w,nperiod+1))    # VALUE FUNCTION
Va_urban        = 1e-8*np.ones((N_a,N_z,N_w,nperiod))       # ENVELOPE CONDITION
cf_urban        = 1e-8*np.ones((N_a,N_z,N_w,nperiod))       # POLICY FUNCTION - FOOD CONSUMPTION
cm_urban        = 1e-8*np.ones((N_a,N_z,N_w,nperiod))      # POLICY FUNCTION - MANUF. CONSUMPTION
a1_urban        = 1e-8*np.ones((N_a,N_z,N_w,nperiod))      # POLICY FUNCTION - SAVING/BORROWING
y_urban         = 1e-8*np.ones((N_a,N_z,N_w,nperiod))      # INCOME
budget        = 1e-8*np.ones((N_a,N_z,N_w,nperiod))
cf_temp         = 1e-8*np.ones((N_a,N_z,N_w))
cm_temp         = 1e-8*np.ones((N_a,N_z,N_w))
a0_temp         = 1e-8*np.ones((N_a,N_z,N_w))  
EV1_temp        = 1e-8*np.ones((N_a,N_z,N_w))  

egm_urban_func = cp.egm_urban_func
interp_kind = 'linear'
minimize_method = 'L-BFGS-B'   # Options: L-BFGS-B, TNC, SLSQP

# with slsqp and linear took 8.9 seconds. A lot of time in first iteration.
# with tnc and linear took 8.9 seconds. A lot of time in first iteration.
#with LBGGSB took 11 sec.
# I choose to use SLSQP

#SLSQP with cubic takes 8.7 sec.
 #SLSQP with nearest takes 7.6 sec but weird shapes!!
 


# EVALUATE THE VALUE AND POLICY FUNCTION AT EACH POINT
it_0 = 0

qe.tic()
for it in reversed(range(it_0, nperiod)):
    str1 = "period = %d " % (it)
    print(' ')
    print(str1)

    # CURRENT LABOR INCOME
    y_urban[:,:,:,it]   = np.exp(cum_gy_u[it] + W + Z)  
 
    # LAST PERIOD: EXOGENOUS GRID METHOD
    if it==nperiod-1:

        print('NOTE: THE LAST PERIOD TAKES A LITTLE BIT OF TIME ... HANG IN THERE!')

        # BORROWING/SAVING DECISION (NO SAVINGS AT TIME T)
        a1_urban[:,:,:,it]  = 0  
        qe.tic()
        for ia in range(0,N_a):
            for iz in range(0,N_z):
                for iw in range(0,N_w):
            
            # SOLVE FOR cf (NONLINEAR EQUATION)
                    
                    x0= c_bar + 1e-5  
                    LB      = c_bar                                              # LOWER BOUND OF cf
                    UB      = y_urban[ia,iz,iw,it] + (1+cp.rrate)*cp.a_state[ia]       # UPPER BOUND OF cf
            
                    egm_urban = lambda params: cp.egm_urban_func(params,cp.a_state[ia],y_urban[ia,iz,iw,it],a1_urban[ia,iz,iw,it])
                    
                    bounds = ((LB,UB),) 
                    x = minimize(egm_urban, x0, method=minimize_method, bounds=bounds)
                    cf_urban[ia,iz,iw,it]  = x.x
                    
                    # COMPUTE cm FROM BUDGET CONSTRAINT
                    cm_urban[ia,iz,iw,it]  = y_urban[ia,iz,iw,it] + (1+cp.rrate)*cp.a_state[ia] - cp.pf*cf_urban[ia,iz,iw,it] 

            
        
        # EXPECTED VALUE FUNCTION AT TIME T+1 TO CALCULATE CURRENT VALUE FUNCTION LATER ON
        EV1_temp = np.zeros((N_a,N_z,N_w))
        qe.toc()
    # t<T: # ENDOGENOUS GRID METHODS
    else: 
        #print(V_urban[:,9,9,it+1])
        # FOR EACH (a',z, w), OBTAIN cm --> cf --> a
        for ia in range(0,N_a):
            for iz in range(0,N_z):
                for iw in range(0,N_w):
                        
                    # INTERPOLATE VALUE FUNCTION AND ENVELOPE CONDITION AT t+1
                    # GIVEN a(t+1) and w(t)
                    VF      = V_urban[ia,:,iw,it+1] 
                    Va      = Va_urban[ia,:,iw,it+1] 
            
                    # EXPECTED VALUE FUNCTION AND ENVELOPE CONDITION
                    EVF    = z_prob[iz,:]@VF  
                    EVa    = z_prob[iz,:]@Va  
                                    
                    # EXPECTED VALUE FUNCTION AT TIME t+1
                    EV1_temp[ia,iz,iw]   = EVF  

                # COMPUTE cm FROM INTERTEMPORAL OPTIMAL CONDITION
                    cm_temp[ia,iz,iw]   = (cp.β*EVa)**(-1/cp.η_2)  
            
                    # COMPUTE cf FROM INTRATEMPORAL OPTIMAL CONDITION
                    cf_temp[ia,iz,iw]   = (cp.κ*cp.pf*(cm_temp[ia,iz,iw]**(-cp.η_2)))**(-1/cp.η_1) + cp.c_bar  
            
                    # COMPUTE a FROM BUDGET CONSTRAINT
                    a0_temp[ia,iz,iw]  = (cp.pf*cf_temp[ia,iz,iw] + cm_temp[ia,iz,iw] + cp.a_state[ia] - y_urban[ia,iz,iw,it])/(1+cp.rrate)  
            
        
        # USING THE COMPUTED a0 VALUES, INTERPOLATE POLICY FUNCTION 
        # (SINCE INPUT GRIDS a0_temp ARE SCATTERED, WE NEED TO EITHER USE DELAUNEY INTERPOLATION
        #  OR ONE DIMENSIONAL INTERPOLATION FOR ALL (Z,E))
        # DUE TO THE INACCURATE INTERPOLATION OF DELAUNEY, WE CHOOSE THE LATTER.
        for iz in range(0,N_z):
                for iw in range(0,N_w):
                    cf_urban_interp = interp1d(a0_temp[:,iz,iw],cf_temp[:,iz,iw],kind=interp_kind, fill_value='extrapolate')  
                    cm_urban_interp = interp1d(a0_temp[:,iz,iw],cm_temp[:,iz,iw],kind=interp_kind, fill_value='extrapolate')  
                    a1_urban_interp = interp1d(a0_temp[:,iz,iw],A[:,iz,iw],kind=interp_kind, fill_value='extrapolate')  
                    
                    #NO NEED INTERPOLATE ALL 3. WITH 2 AND THEN USE RESIDUAL IS ENOUGH.
                    cf_urban[:,iz,iw,it] = cf_urban_interp(A[:,iz,iw])               
                    cm_urban[:,iz,iw,it] = cm_urban_interp(A[:,iz,iw])
                    a1_urban[:,iz,iw,it] = a1_urban_interp(A[:,iz,iw])
                    
        #cm_urban[:,:,:,it] = ((cf_urban[:,:,:,it]-cp.c_bar)**(-cp.η_1)/(cp.κ*cp.pf))**(-1/cp.η_2)
        # CHECK BUDGET CONSTRAINT. COULD BE A PROBLEM ON THERE.
        a1_temp                 = a1_urban[:,:,:,it]
        index                   = a1_temp < cp.a_bar  

        # IF BORROWING CONSTRAINT IS BINDING
        for ia in range(0,N_a):
            for iz in range(0,N_z):
                for iw in range(0,N_w):
                    if index[ia,iz,iw]==1:

                        # BINDING CONSTRAINT
                        a1_urban[ia,iz,iw,it]   = cp.a_bar  

                        # SOLVE FOR cf (NONLINEAR EQUATION)
                        init_param = cp.c_bar + 1e-5  
                        LB      = c_bar                                          # LOWER BOUND OF cf
                        UB      = y_urban[ia,iz,iw,it] + (1+cp.rrate)*cp.a_state[ia]     # UPPER BOUND OF cf
                        egm_urban = lambda params: cp.egm_urban_func(params,cp.a_state[ia],y_urban[ia,iz,iw,it],cp.a_bar)
                    
                        bounds = ((LB,UB),) 
                        x = minimize(egm_urban, x0, method=minimize_method, bounds=bounds)
                        cf_urban[ia,iz,iw,it]  = x.x

                        # COMPUTE cm FROM BUDGET CONSTRAINT
                        cm_urban[ia,iz,iw,it]  = y_urban[ia,iz,iw,it] + (1+cp.rrate)*cp.a_state[ia] - cp.pf*cf_urban[ia,iz,iw,it] - cp.a_bar  

    budget[:,:,:,it] = 100*((cp.pf*cf_urban[:,:,:,it] +cm_urban[:,:,:,it] +a1_urban[:,:,:,it] -(y_urban[:,:,:,it] + (1+cp.rrate)*cp.A))/y_urban[:,:,:,it])
       
    # COMPUTE VALUE FUNCTION AND ENVELOPE CONDITION FOR NEXT ITERATION
    V_urban[:,:,:,it]  = ((cf_urban[:,:,:,it]-cp.c_bar)**(1-cp.η_1))/(1-cp.η_1) + cp.κ*(cm_urban[:,:,:,it]**(1-cp.η_2))/(1-cp.η_2) + cp.β*EV1_temp  
    Va_urban[:,:,:,it] = (1+cp.rrate)*cm_urban[:,:,:,it]**(-cp.η_2)  
qe.toc()

#Plot optimal policies   
 #%% 
plot = True

if plot == True:
    w_0 = 0
    save = True
    legend = True
    
    # Food Consumption policy
    for it in range(it_0,8):
        cp.plot_policies( cf_urban,'Food Consumption ', it=it, fixi=w_0,legend=legend, folder=folder, save=save)
    
    # Manufacturing consumption policy
    for it in range(it_0,8):
        cp.plot_policies( cm_urban,'Manufacturing Good Consumption ',fixi=w_0,legend=legend, folder=folder, it=it, save=save)
    
    #Plot Assets policy
    for it in range(it_0,8):
        cp.plot_policies( a1_urban,'Asset policy ', it=it,fixi=w_0,legend=legend, line_45=True, folder=folder, save=save)

    #Plot EVk:    
    for it in range(it_0,8):
        cp.plot_policies( V_urban,'Value Function ', it=it,fixi=w_0,legend=legend, folder=folder, save=save)   

    #Plot EVk:    
    for it in range(it_0,8):
        cp.plot_policies( Va_urban,'Envelope condition', it=it,fixi=w_0,legend=legend, folder=folder, save=save)   
        
    #Output   
    for it in range(it_0,8):
        cp.plot_policies( y_urban,'Output ', it=it, fixi=w_0,legend=legend, folder=folder, save=save)
     
    #Budget Constraint: Should be line on 0.
    for it in range(it_0,8):
        cp.plot_policies( budget,'Budget as % output ', it=it,legend=legend, fixi=w_0, folder=folder, save=save)
        
 
#%% Save policy functions
        
save_policies = True

if save_policies == True:
    np.save(folder+'cf_urban', cf_urban)
    np.save(folder+'cm_urban', cm_urban)
    np.save(folder+'a1_urban', a1_urban)
    np.save(folder+'V_urban', V_urban)


'''
#% simulation

#k_sim, z_sim, w_sim = cp.simulate_distr(k1_rural, N=1000)



#def simulate_distr(self, k_policy, N_sim=1000):



    ## #  SIMULATION

N_sim = 10000
#  SIMULATED DECISIONS
CF_SIM     =  1e-8*np.ones((N_sim,nperiod))        #  BEFORE SUBSIDY FOOD CONSUMPTION
CM_SIM     =  1e-8*np.ones((N_sim,nperiod))        #  NONFOOD CONSUMPTION
Y_SIM      =  1e-8*np.ones((N_sim,nperiod))        #  BEFORE TAX INCOME
A_SIM      =  1e-8*np.ones((N_sim,nperiod+1))      #  ASSETS
Z_SIM      =  1e-8*np.ones((N_sim,nperiod))        #  PERMANENT SHOCKS
W_SIM      =  1e-8*np.ones((N_sim,1))              #  INITIAL PRODUCTIVITY
        #  EX-ANTE TAX
FISHY_SIM  =  1e-8*np.ones((N_sim,nperiod))        #  FISHY SIMULATION
a_mean = 1e-8*np.ones((nperiod))
cf_mean = 1e-8*np.ones((nperiod))
cm_mean = 1e-8*np.ones((nperiod))
y_mean = 1e-8*np.ones((nperiod))
a_var = 1e-8*np.ones((nperiod))
cf_var = 1e-8*np.ones((nperiod))
cm_var = 1e-8*np.ones((nperiod))
y_var = 1e-8*np.ones((nperiod))





#  SAMPLING FROM MULTIVARIATE NORMAL DISTRIBUTION
MU              = [cp.mean_lnw_u, cp.mean_lna0_u]         #  MEAN OF JOINT DISTRIBUTION
SIG             = np.array([cp.var_lnw_u,0,0, cp.var_lna0_u]).reshape(2,2)         #  VAR-COV MATRIX
INIT            = np.random.multivariate_normal(MU,SIG,N_sim)              #  RANDOM GENERATOR


#  PRODUCTIVITY SHOCK
z_eps           = np.random.normal(0,cp.sig_z,[N_sim,nperiod]) 

#  INITIAL HETEROGENEITY
A_SIM[:,0]      = np.exp(INIT[:,1])             #  INITIAL CAPITAL
W_SIM[:,0]      = INIT[:,0]                     #  INITIAL PRODUCTIVITY
Z_SIM[:,0]      = 0                             #  NO INITIAL PRODUCTIVITY SHOCK

# TRIM INITIAL PRODUCTIVITY TO THE LOWER BOUND SO THAT EXTRAPOLATION WILL
# BE PREVENTED.
ind_w       = (W_SIM[:,0]<cp.w_min)
W_SIM[ind_w,0] = cp.w_min

data_linear = np.column_stack((A.ravel(1), Z.ravel(1), W.ravel(1)))


for it in range(0,nperiod):

    Y_SIM[:,it]     = np.exp(cum_gy_u[it] + W_SIM[:,0] + Z_SIM[:,it])
   
    #  INTERPOLATE DECISIONS FROM POLICY FUNCTIONS
    #  DUE TO THE INACCURACY OF INTERPOLATION, THE BUDGET CONSTRAINT MAY NOT HOLD IN EQUALITY
    rbf_a    = my_interpolation(data_linear,a1_urban[:,:,:,it].ravel(1)) 
    rbf_cf    = my_interpolation(data_linear,cf_urban[:,:,:,it].ravel(1)) 
    #rbf_cm    = my_interpolation(data_linear,cm_urban[:,:,:,it].ravel(1), fill_value = np.mean(cm_urban[:,:,:,it])) 
    
    #rbf_a    = my_interpolation(data_linear,a1_urban[:,:,:,it].ravel(1), fill_value = cp.a_bar) 
    #rbf_cf    = my_interpolation(data_linear,cf_urban[:,:,:,it].ravel(1), fill_value =cp.c_bar ) 
    
    
    #rbf_cm    = my_interpolation(data_linear,cm_urban[:,:,:,it].ravel(1), fill_value = 0) 
    
    
    A_SIM[:,it+1]   = rbf_a(Z_SIM[:,it],A_SIM[:,it],W_SIM[:,0])
    CF_SIM[:,it]    = rbf_cf(Z_SIM[:,it],A_SIM[:,it],W_SIM[:,0]) 
    #CM_SIM[:,it]    = rbf_cm(Z_SIM[:,it],A_SIM[:,it],W_SIM[:,0]) 
    CM_SIM[:,it] = ((CF_SIM[:,it]-cp.c_bar)**(-cp.η_1)/(cp.κ*cp.pf))**(-1/cp.η_2)  
            



    #  NEXT PERIOD'S PRODUCTIVITY
    if it<nperiod-1:    
        Z_SIM[:,it+1]      = cp.ρ*Z_SIM[:,it] + z_eps[:,it+1]
    

    for n in range(0,N_sim):      
        if A_SIM[n,it]<0 or np.isnan(K_SIM[n,it]) or np.isinf(K_SIM[n,it]):
            A_SIM[n,it] = np.nan
        if CF_SIM[n,it]<c_bar or CM_SIM[n,it]<0:
            
    A_SIM[:,it] = A_SIM[:,it][~np.isnan(A_SIM[:,it])]
    CF_SIM[:,it] = CF_SIM[:,it][~np.isnan(CF_SIM[:,it])]
    CM_SIM[:,it] = CM_SIM[:,it][~np.isnan(CM_SIM[:,it])]
    
    
    #First Moment
    a_mean[it] = np.mean(A_SIM[:,it])
    cf_mean[it] = np.mean(CF_SIM[:,it])
    cm_mean[it] = np.mean(CM_SIM[:,it])
    y_mean[it] = np.mean(Y_SIM[:,it])
    

    
    #Second Moment
    a_var[it] = np.var(A_SIM[:,it])
    cf_var[it] = np.var(CF_SIM[:,it])
    cm_var[it] = np.var(CM_SIM[:,it])
    y_var[it] = np.var(Y_SIM[:,it])


#%% Plots

folder = 'C:/Users/rodri/Dropbox/Eating UP Productivity/Model/python_Albert/figures Nk=50, Nz=3, Nw=3/urban/'
    
save = True

cp.plot_sim(a_mean, name=' Mean of assets', folder=folder, save=save)
cp.plot_sim(a_var, name=' Variance of assets',folder=folder, save=save)

cp.plot_sim(y_mean, name=' Mean of Output',folder=folder, save=save)
cp.plot_sim(y_var, name=' Variance of Output', folder=folder, save=save)

    
            
cp.plot_sim(cf_mean, ' Mean of Food Consumption', folder=folder, save=save)
cp.plot_sim(cf_var, ' Variance of Food Consumption', folder=folder, save=save)

cp.plot_sim(cm_mean, name=' Mean of Cm', folder=folder, save=save)
cp.plot_sim(cm_var, name=' Variance of Cm', folder=folder, save=save)      
''' 