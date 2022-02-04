# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:12:34 2017

@author: Albert
"""

# =============================================================================
#  PLOTTING tutorial
# =============================================================================

import numpy as np

# Matplotlib: Default plotting in python.
import matplotlib.pyplot as plt

# Seaborn: Tries to improve and enhance matplotlib. Specially good for data visualization or complex plots
import seaborn as sns


# =============================================================================
# Matplotlib examples
# =============================================================================
#Matlab-style
#ex1:
x=np.linspace(0,10,200)
y=np.sin(x)
plt.plot(x,y,'b-', linewidth=2)
plt.show()


#Python style: Object-oriented programming
fig,ax = plt.subplots()    #We identify the object
x = np.linspace(0,10,200)
y = np.sin(x)
ax.plot(x,y,'b-',label='sine function', linewidth=2)
ax.legend(loc='upper center')
#fig.savefig('blublu.png')
plt.show()

#We can also add LaTex in a simple way (like in Matlab)
fig,ax = plt.subplots()    #We identify the object
x = np.linspace(0,10,200)
y = np.sin(x)
ax.plot(x, y, 'r-', label=r'$y=\sin(x)$', linewidth=2)   #r in the label for raw string
ax.legend(loc='upper center')
plt.show()

#Controlling ticks, add titles, and so on is also easy:
 fig,ax = plt.subplots()    #We identify the object
x = np.linspace(0,10,200)
y = np.sin(x)
ax.plot(x, y, 'r-', label=r'$y=\sin(x)$', linewidth=2)   #r in the label for raw string
ax.legend(loc='upper center')
ax.set_yticks([-1,0,1])
ax.set_title('test plot')
plt.show()


# Multiple plots in the same graph
from scipy.stats import norm
from random import uniform

fig,ax = plt.subplots()
x=np.linspace(-4,4,150)
for i in range(3):
    m,s = uniform(-1,1), uniform(1,2)
    y=norm.pdf(x, loc=m, scale=s)
    current_label = r'$\mu = {0:2f}$'.format(m)
    ax.plot(x, y, linewidth=2,label=current_label)
ax.legend()
plt.show()


#Multiple subplots (graph)
num_rows, num_cols = 3, 2
fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 12)) #we fix the number and order of graphs here
for i in range(num_rows):
    for j in range(num_cols):
        m, s = uniform(-1, 1), uniform(1, 2)
        x = norm.rvs(loc=m, scale=s, size=100)
        axes[i, j].hist(x, alpha=0.6, bins=20)
        t = r'$\mu = {0:.1f}, \quad \sigma = {1:.1f}$'.format(m, s)
        axes[i, j].set_title(t)
        axes[i, j].set_xticks([-4, 0, 4])
        axes[i, j].set_yticks([])
plt.show()



# 3D Graphs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from matplotlib import cm

def f(x, y):
    return np.cos(x**2 + y**2) / (1 + x**2 + y**2)

xgrid = np.linspace(-3, 3, 50)
ygrid = xgrid
x, y = np.meshgrid(xgrid, ygrid)  

fig = plt.figure(figsize=(8, 6)) #meshgrid + plt.figure
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,
                y,
                f(x, y),
                rstride=2, cstride=2,
                cmap=cm.jet,
                alpha=0.7,
                linewidth=0.25)
ax.set_zlim(-0.5, 1.0)
plt.show()


#Customizing the plots

#We can customize some plots that we regularly use. For example, we can costumize one where the origin appears and other things.

def subplots():
    "Custom subplots with axes throught the origin"
    fig, ax = plt.subplots()

    # Set the axes through the origin
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_position('zero')
    for spine in ['right', 'top']:
        ax.spines[spine].set_color('none')

    ax.grid()
    return fig, ax


fig, ax = subplots()  # Call the local version, not plt.subplots()
x = np.linspace(-2, 10, 200)
y = np.sin(x)
ax.plot(x, y, 'r-', linewidth=2, label='sine function', alpha=0.6)
ax.legend(loc='lower right')
plt.show()



# =============================================================================
# SEABORN examples
# =============================================================================


##  SEE THE EXAMPLE CODE OF macro_evidence_uganda.py TO LOOK AT EXAMPLES OF SEABORN PLOTS.









