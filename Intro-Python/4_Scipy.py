# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:48:59 2017

@author: Albert
"""

# =============================================================================
# SCIPY TUTORIAL
# =============================================================================
#Scipy builds on top of numpy to provide common tools for scientific programming such as:
#Linear algebra, numerical integration, interpolation, optimization, 
#distribution and random numbers gen, signal processing, etc
    

import numpy as np


        
### ROOT FINDING ---------------------------------------
f = lambda x: np.sin(4 * (x - 0.25)) + x + x**20 - 1

#We also have a build-in function for bisection method:
from scipy.optimize import newton
newton (f,0.2)
newton(f,0.7)  

#Hybrid methods
"""
     hybrid methods typically combine a fast method with a robust method st:
         1. Attempt to use a fast method
         2. Check diagnostics
         3. If diagnosis are bad, switch to more robust one.
"""
    
#Example of hybrid: brenqt
from scipy.optimize import brentq
brentq(f,0,1)
%timeit brentq(f,0,1)  #We obtain the correct solution at almost the same speed.


#FIXED POINTS (f(x*)=x*) ------------------------------
from scipy.optimize import fixed_point
fixed_point(lambda x: x**2,10.0) #10.0 is the initial guess.



# OPTIMIZATION --------------------------------------
  """
 Most packages provide only functions for minimization
 Minimization is closely related to root finding and the trade-off btw speed-robustness is also present.
 Unless we have prior info to exploit, it is best to use Hybrid methods.
 """
 
#For constrained, univariate minimization, a good hybrid option is:
from scipy.optimize import fminbound
fminbound(lambda x: x**2,-1,2) #search in [-1,2]

#Alternatively
def y(x):
    return x**2


fminbound(y,-1,2)

 # Multivariate local optimizers include:
     """
     minimize
     fmin
     fmin_powell
     fmin_cg
     fmin_bgfs
     fmin_ncg
     """
     
 # Constrained multivariate local optimizers include:
   """
   fmin_l_bfgs_b
   fmin_tnc
   fmin_cobyla
   """

## CHECK SCIPY DOCUMENTATION
   
# INTEGRATION -------------------------------------
   """
   Most numerical integration methods work by computing the integral of an approximating polynomial.
   In scipy the rellevant module is: scipy.integrate
   
   A good default for univariate integration is quad:
       quad uses Clenshaw-Curtis quadrature, based on expansion in terms of Chebychev polynomials.
       A useful alternative for univariate integration is fixed_quad which is faster and therefore better for loops.
       
   There are also functions for multivariate integration.
    """

from scipy.integrate import quad
integral, error = quad(lambda x: x**2,0,1)
























































