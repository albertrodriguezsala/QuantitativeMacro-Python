
# coding: utf-8

# Authors: Albert Rodriguez 
# 

# =============================================================================
# Approximating the CES function
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set some parameters
alpha = 0.5
sigma = 0.25
k_min = 1e-3
k_max = 10
h_min = 1e-3
h_max = 10
k = np.linspace(0.1,10,100)
h = np.linspace(0.1,10,100)
a = 0
b = 10
n = 20
order = 3

# Define CES function
def y_fun(k,h):
    ces = ((1-alpha)*k**((sigma-1)/sigma)+alpha*h**((sigma-1)/sigma))**(sigma/(sigma-1))
    return ces

y = y_fun(k,h)
y_real = np.matrix(y)


# Step 1 & Step 2: Construct Chebyshev nodes-----------------------
def cheb_nodes2d(n,a,b,c,d):
    # n: grid size
    # a,b: upper and lower bound 1st dimension
    # c,d: upper and lower bound 2nd dimension
    x = []
    y = []
    z = []
    for j in range(1,n+1):   
        z_k=-np.cos(np.pi*(2*j-1)/(2*n))   ## Get the Chebyshed node in [-1,1]
        x_j=(z_k+1)*((b-a)/2)+a  ## Convert z_point to x dimension
        y_j=(z_k+1)*((d-c)/2)+c  ## convert z_point to y dimension
        z.append(z_k)
        x.append(x_j)
        y.append(y_j)
    return (np.array(z),np.array(x),np.array(y))

z, k_nodes, h_nodes = cheb_nodes2d(n,a,b,a,b)
   
 
# Step 3: Grid of the images of the nodes-----------------------
y_nodes = np.matrix(y_fun(k_nodes[:,None],h_nodes[None,:]))


# Step 4: Create the Chebyshev basis functions -------------------
def T(d,x):
    #d: Degree level
    #x: Chebyshev node
    psi = []
    psi.append(np.ones(len(x)))
    psi.append(x)
    for i in range(1,d):
        p = 2*x*psi[i]-psi[i-1]
        psi.append(p)
    key = np.matrix(psi[d]) 
    return key   


### Step 5: Obtain coefficients running OLS -------------------------
def coeff(z,y_nodes,d):
    theta=np.empty((d+1)*(d+1))
    theta.shape = (d+1,d+1)
    for i in range(d+1):  ##loop over first dimension
        for j in range(d+1):  ##loop over second dimension
            theta[i,j] = (np.sum(np.array(y_nodes)*np.array((np.dot(T(i,z).T,T(j,z)))))/np.array((T(i,z)*T(i,z).T)*(T(j,z)*T(j,z).T)))
    return theta

theta = coeff(z,y_nodes,order)

# Step 5: Obtain approximation function ---------------------------------
def f_approx(x,y,theta,d):
    f = []
    in1 = (2*(x-a)/(b-a)-1)
    in2 = (2*(y-a)/(b-a)-1)
    for u in range(d):
        for v in range(d):
                f.append(np.array(theta[u,v])*np.array((np.dot(T(u,in1).T,T(v,in2)))))
    f_sum = sum(f)
    return f_sum

y_approx = f_approx(k,h,theta,order)
y_real = y_fun(k[:,None],h[None,:])
error = y_real - y_approx



# Plotting --------------------------------------------------

### Function vs approximation

def surface_plot_real_vs_approx(z_real, z_approx, x, y, title ='Figure Real vs Approx'):
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot( projection='3d')
    X, Y = np.meshgrid(x,y)
    ax1.plot_surface(X,Y,z_real)
    ax1.plot_surface(X,Y,z_approx)
    ax1.set(title=title +'order'+str(order))
    return plt.show()

surface_plot_real_vs_approx(y_real,y_approx, k, h)


### error plot
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
K, H = np.meshgrid(k,h)
ax1.plot_surface(K,H,error)
ax1.set(title='Error aproximation of order ' + str(order))


### contour plot
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
CS = ax2.contour(K,H,error)
plt.clabel(CS, inline=1, fontsize=10)
plt.show()




#%% Order 5

order = 5
theta = coeff(z,y_nodes,order)
y_approx = f_approx(k,h,theta,order)
error = y_real - y_approx

surface_plot_real_vs_approx(y_real,y_approx, k, h)


# Plotting
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
K, H = np.meshgrid(k,h)
ax1.plot_surface(K,H,error)
ax1.set(title='Error aproximation of order ' + str(order))
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
CS = ax2.contour(K,H,error)
plt.clabel(CS, inline=1, fontsize=10)
plt.show()


#%% Order 10

order = 10
theta = coeff(z,y_nodes,order)
y_approx = f_approx(k,h,theta,order)
error = y_real - y_approx

surface_plot_real_vs_approx(y_real,y_approx, k, h)



# Plotting
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
K, H = np.meshgrid(k,h)
ax1.plot_surface(K,H,error)
ax1.set(title='Error aproximation of order ' + str(order))
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
CS = ax2.contour(K,H,error)
plt.clabel(CS, inline=1, fontsize=10)
plt.show()


# In[4]:

order = 15
theta = coeff(z,y_nodes,order)
y_approx = f_approx(k,h,theta,order)
error = y_real - y_approx

surface_plot_real_vs_approx(y_real,y_approx, k, h)


# Plotting
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
K, H = np.meshgrid(k,h)
ax1.plot_surface(K,H,error)
ax1.set(title='Error aproximation of order ' + str(order))
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
CS = ax2.contour(K,H,error)
plt.clabel(CS, inline=1, fontsize=10)
plt.show()

