# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 12:46:32 2017

@author: Albert
"""

#Object Oriented Programming
   """
   Python supports both procedural and object-oriented programming. Matlab and fortram are mainly procedural while C++ is procedural with OOP added on top.
   The procedural paradigm works as follows:
       Program has a state that contain values of its variables
       Fuctions are called to act on these data
       data are passed back via function calls.
   In contrast, in the OOP paradigm data and functions are BUNDLED TOGETHER into objects. 
      In the OOP functions are usually called methods.
      In python the data andm ethods of an object are referred to as attributes.
      Depending on the class of object(like float, inter, str, etc) the object will have different methods.
   """
#In procedural programming, a list will be treated as data. However, in python it is treated as an object so it has methods that can be applied on it.

x= [1,5,4]
x.sort()
x.__class__
dir(x)  #To obtain the list of all the attributes or methods that the object have.
#OOP is useful as for the same reason abstraction is: for recognizing and exploiting common structure.


#Example: defining our own class:
class Consumer:
    pass

c1=Consumer()  #Create an instance/object.
c1.wealth = 10
c1.wealth

#however we didn't define methods for that class. Let's work on a more complete example.
#EX: Consumer class with more structure: A wealth attribute, an earn(Y) method. A spend method where spend(x) either deceases wealth by x. this type of class allow to internalize some new syntax.
class Consumer:
    
    def __init__(self, w):
        "Initialize consumer with w dollars of wealth"
        self.wealth = w
        
    def earn(self, y):
        "The consumer earns y dollars" 
        self.wealth += y
        
    def spend(self, x):
        "The consumer spends x dollars if feasible"
        new_wealth = self.wealth - x
        if new_wealth < 0:
            print("Insufficent funds")
        else:
            self.wealth = new_wealth
            
#Explanation:
    """
    This class defines instance data (wealth) and three methods:__init__, earn, spend.
    Wealth is instance data because each consumer we create(each instante of the consumer class) will have its own separate wealth data.
     __init__ method is a constructor method. whenever we create an instance of the class, this method will be called automatically.
    """
# Usage:
    c1= Consumer(10)
    c1.spend(5)
    c1.wealth
    
    c1.earn(15)
    c1.spend(100)
    
#We can create multiple instances each with its own data:
    c1 = Consumer(10)
    c2 = Consumer(12)
    c2.spend(4)
    c2.wealth

#In fact each instance stores its data in a separate namespace dictionary:
    c1.__dict__
    c2.__dict__
    
# SELF: "It is used to identify the object that is created".
     """
     Any instance data should be prepended with self. 
     Any method defined within the class should have self as its first argument.
     Any method referenced within the class should be called as self.method_name
     """









# EXAMPLE: THE SOLOW MODEL
import numpy as np

class Solow:
    r"""
    Implements the Solow growth model with update rule

    .. math::
        k_{t+1} = \frac{s z k^{\alpha}_t}{1 + n}  + k_t \frac{1 - d}{1 + n}
    
    """

    def __init__(self, n, s, d, alpha, z, k):
        """ 
        Solow growth model with Cobb Douglas production function.  All
        parameters are scalars.  See http://quant-econ.net/py/python_oop.html
        for interpretation.
        """
        self.n, self.s, self.d, self.alpha, self.z = n, s, d, alpha, z
        self.k = k
        

    def h(self):
        "Evaluate the h function"
        temp = self.s * self.z * self.k**self.alpha + self.k * (1 - self.d)
        return temp / (1 + self.n)

    def update(self):
        "Update the current state (i.e., the capital stock)."
        self.k =  self.h()
        
    def steady_state(self):
         "Compute the steady state value of capital."
         return ((self.s * self.z) / (self.n + self.d))**(1 / (1 - self.alpha))
     
    def generate_sequence(self, t):
        "Generate and return a time series of length t"
        path = []
        for i in range(t):
            path.append(self.k)
            self.update()
        return path
    
import matplotlib.pyplot as plt

baseline_params = 0.05, 0.25, 0.1, 0.3, 2.0, 1.0
s1 = Solow(*baseline_params)  # The 'splat' operator * breaks up the tuple
s2 = Solow(*baseline_params)
s2.k = 8.0  # Reset s2.k to make high capital economy
T = 60
fig, ax = plt.subplots()
# Plot the common steady state value of capital
ax.plot([s1.steady_state()]*T, 'k-', label='steady state')
# Plot time series for each economy
for s in s1, s2:
    lb = 'capital series from initial state {}'.format(s.k)
    ax.plot(s.generate_sequence(T), 'o-', lw=2, alpha=0.6, label=lb)

ax.legend(loc='lower right')
plt.show()



#EXAMPLE 2: A SIMPLE MARKET
from scipy.integrate import quad

class Market:

    def __init__(self, ad, bd, az, bz, tax):
        """
        Set up market parameters.  All parameters are scalars.  See
        http://quant-econ.net/py/python_oop.html for interpretation.

        """
        self.ad, self.bd, self.az, self.bz, self.tax = ad, bd, az, bz, tax
        if ad < az:
            raise ValueError('Insufficient demand.')
        
    def price(self):
        "Return equilibrium price"
        return  (self.ad - self.az + self.bz*self.tax)/(self.bd + self.bz) 
    
    def quantity(self):
        "Compute equilibrium quantity"
        return  self.ad - self.bd * self.price()
        
    def consumer_surp(self):
        "Compute consumer surplus"
        # == Compute area under inverse demand function == #
        integrand = lambda x: (self.ad/self.bd) - (1/self.bd)* x
        area, error = quad(integrand, 0, self.quantity())
        return area - self.price() * self.quantity()
    
    def producer_surp(self):
        "Compute producer surplus"
        #  == Compute area above inverse supply curve, excluding tax == #
        integrand = lambda x: -(self.az/self.bz) + (1/self.bz) * x
        area, error = quad(integrand, 0, self.quantity())  
        return (self.price() - self.tax) * self.quantity() - area
    
    def taxrev(self):
        "Compute tax revenue"
        return self.tax * self.quantity()
        
    def inverse_demand(self,x):
        "Compute inverse demand"
        return self.ad/self.bd - (1/self.bd)* x
    
    def inverse_supply(self,x):
        "Compute inverse supply curve"
        return -(self.az/self.bz) + (1/self.bz) * x + self.tax
    
    def inverse_supply_no_tax(self,x):
        "Compute inverse supply curve without tax"
        return -(self.az/self.bz) + (1/self.bz) * x

#Check:
    baseline_params = 15, .5, -2, .5, 3
    m = Market(*baseline_params)
    print("equilibrium price = ", m.price())
    
#And the plot:
import matplotlib.pyplot as plt
import numpy as np

# Baseline ad, bd, az, bz, tax
baseline_params = 15, .5, -2, .5, 3
m = Market(*baseline_params) 

q_max = m.quantity() * 2
q_grid = np.linspace(0.0, q_max, 100)
pd = m.inverse_demand(q_grid)
ps = m.inverse_supply(q_grid)
psno = m.inverse_supply_no_tax(q_grid)    

fig, ax = plt.subplots()
ax.plot(q_grid, pd, lw=2, alpha=0.6, label='demand')
ax.plot(q_grid, ps, lw=2, alpha=0.6, label='supply') 
ax.plot(q_grid, psno, '--k', lw=2, alpha=0.6, label='supply without tax')
ax.set_xlabel('quantity', fontsize=14)
ax.set_xlim(0, q_max)
ax.set_ylabel('price', fontsize=14)
ax.legend(loc='lower right', frameon=False, fontsize=14)
plt.show()


def deadw(m):
    "Computes deadweight loss for market m."
    # == Create analogous market with no tax == #
    m_no_tax = Market(m.ad, m.bd, m.az, m.bz, 0)   
    # == Compare surplus, return difference == #
    surp1 = m_no_tax.consumer_surp() + m_no_tax.producer_surp()  
    surp2 = m.consumer_surp() + m.producer_surp() + m.taxrev()
    return surp1 - surp2

baseline_params = 15, .5, -2, .5, 3
m = Market(*baseline_params)
deadw(m)  # Show deadweight loss



#If you want to provide a return value for the len function when applied to your user-defined object, use the __len__ special method

#EXERCISE 1

class ECDF:
   def __init__(self,observations):
       self.observations=observations
       
   def __call__(self,x):
       count=0.0
       for obs in self.observations:
           if obs<=x:
               count+=1
       return count / len(self.observations)
           
 #check
from random import uniform 
sample = [uniform(0,1) for i in range(10)]
f= ECDF(sample) 
print(f(0.5))

f.observations =  [uniform(0,1) for i in range(10)]
print(f(0.5))


#EXERCISE 2

class Polynomial:
    def __init__(self, coefficients):
        """
        Creates an instance of the Polynomial class representing

            p(x) = a_0 x^0 + ... + a_N x^N,

        where a_i = coefficients[i].
        """
        self.coefficients=coefficients
        
    def __call__(self, x):
        y=0
        for i, a in enumerate(self.coefficients):  #Important to use ennumerate if we want to index!!
            y += a*x**i    #augmented assignment: y+=1 equivalent to y=y+1 but y is only evaluated once.
        return y
    
    def differenciate(self): #differenciate will be a method that we will call with f.differenciate
        new_coefficients=[]
        for i,a in enumerate(self.coefficients):
            new_coefficients.append(a*i)
        del new_coefficients[0]
        self.coefficients = new_coefficients
        return new_coefficients
     
coeff= [1,0,0,1]
f_poly=Polynomial(coeff)
print(f_poly(2))






