""" Code for simple GP regression. It assumes a zero mean GP Prior. Kernel parameter optimization is performed using gradient ascend and active learning based on GP is implemented """

from __future__ import division
import numpy as np
import matplotlib.pyplot as pl

# This is the true unknown function we are trying to approximate
f = lambda x: np.sin(0.9*x+ 0.4).flatten()
# f = lambda x: (0.25*(x**2) - 0.1*(x**3) + x).flatten()

# Define the kernel
def kernel(a, b,l=0.1):
    """ GP squared exponential kernel """
    # Same as RBF kernel
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/l) * sqdist)

# Function for find the best kernel parameter
def gradient_ascend_for_kernel_parameters(X,y,kernelParameter = 0.1):
    iterations = 2000
    K = kernel(X, X, kernelParameter)
    n = K.shape[0]
    K = K + 1e-6*np.eye(n)
    n = K.shape[0]
    step_length = 0.001
    sqdist = np.sum(X**2,1).reshape(-1,1) + np.sum(X**2,1) - 2*np.dot(X, X.T)

    for i in range(0,iterations):
        dKdl = 0.5 * (K- 1e-6*np.eye(n))* sqdist/kernelParameter**2 
        L = np.linalg.cholesky(K) 
        L_inverse =  np.linalg.inv(L)
        K_inverse = np.dot(L_inverse.T,L_inverse)
        alpha  = np.dot(K_inverse,y).reshape(-1,1)
        A = np.dot(alpha,alpha.T) - K_inverse
        B = np.dot(A,dKdl)
        gradient = 0.5 * np.sum(np.diag(B),0)
        l = kernelParameter + step_length*gradient
        if l < 1e-2:
            l = 1e-2
        if l > 100:
            l = 100
        K = kernel(X, X, l) + 1e-6*np.eye(n)
        kernelParameter = l

    print(f"iteration {i + 1}: kernelParameter = {kernelParameter}")
    return kernelParameter    

    
""" To draw samples from the posterior distribution"""

d = 1
N = 5         # number of training points.
n = 100         # number of test points.
sy = 0.001    # noise variance.
span = 20

# Sample some input points and noisy versions of the function evaluated at
# these points. 
X = np.random.uniform(-span, span, size=(N,d))
y = f(X) + sy*np.random.randn(N)

kernelParameter = gradient_ascend_for_kernel_parameters(X,y)
K = kernel(X, X,kernelParameter)
K = K + sy*np.eye(N)
# noicy gaussian process, variance s at each data point observation, they are independent on each other.
L = np.linalg.cholesky(K)  

# points we're going to make predictions at.
Xtest = np.linspace(-span, span, n).reshape(-1,1) 
Lk = np.linalg.solve(L, kernel(X, Xtest,kernelParameter))  
# here we calculate kx*.T L^-T, solve x for Lx = y, get L^-1 y, then take the dot product.
mu = np.dot(Lk.T, np.linalg.solve(L, y))    
# compute the variance at our test points.
# calculate k**
K_ = kernel(Xtest, Xtest,kernelParameter)          
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)    
# this is the standard deviation.
s = np.sqrt(s2)                             

# PLOTS:
pl.figure(1)
pl.clf()
pl.plot(X, y, 'r+', ms=20)
pl.plot(Xtest, f(Xtest), 'b-')
pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
print(f"mu {mu.shape} Xtest {Xtest.shape} X {X.shape} y {y.shape}")
pl.plot(np.block(Xtest), np.block(mu.reshape((-1,1))).flat, 'r--', lw=2)
pl.savefig('predictive.png', bbox_inches='tight')
pl.title('Mean predictions plus 3 st.deviations')
pl.axis([-span, span, -5, 5])

# draw samples from the prior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
f_prior = np.dot(L, np.random.normal(size=(n,10)))
# this is due to the fact that every multivariate gaussian distribution can be written as X = AZ + mu, Z is a vector of n independent standard normal distribution, and A A.T = K, now L is a perfect A, so we can first sample from Z then do linear transformation to obtain X.
pl.figure(2)
pl.clf()
pl.plot(Xtest, f_prior)
pl.title('Ten samples from the GP prior')
pl.axis([-span, span, -5, 5])
pl.savefig('prior.png', bbox_inches='tight')

# draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,10)))
pl.figure(3)
pl.clf()
pl.plot(Xtest, f_post)
pl.title('Ten samples from the GP posterior')
pl.axis([-span, span, -5, 5])
pl.savefig('post.png', bbox_inches='tight')

pl.show()

""" Bayesian optimization (Active learning) based on GP """
# find the best point x_next to query
# define utility function to trade off exploration and exploitation  U = mu(x) + C sigma(x)
C = 10
step = 0.1
limit = int(input("how many points to query:"))
L = np.linalg.cholesky(K)  
for point in range(0,limit):
    x_next = np.random.uniform(span, -span, size=(1,d))
    L_inverse =  np.linalg.inv(L)
    K_inverse = np.dot(L_inverse.T,L_inverse)
    for t in range(0,1000):
        X_diff = X - x_next
        k_x_star = kernel(X,x_next,kernelParameter)
        n_points = X.shape[0]
        dummy =  np.zeros((n_points,n_points))
        np.fill_diagonal(dummy,k_x_star)
        J_dk_x_star_dx_T = np.dot(X_diff.T,dummy)
        grad = np.dot(np.dot(J_dk_x_star_dx_T,K_inverse),y.reshape((-1,1))-2*C*k_x_star)        
        x_next = x_next + step*grad

    if x_next > span:
        x_next = span - np.random.uniform(0,span,size=(1,1))
    if x_next < -span:
        x_next = -span + np.random.uniform(0,span,size=(1,1))
    
    print(f"best next point {x_next}")
    y_next = f(np.array(x_next)) + sy*np.random.randn(1)
    X = np.block([[X],[x_next]])
    y = np.block([[y.reshape(-1,1)],[y_next]])  
    
# Here for each time we query a new point, the kernel 
# parameter is reoptimized, therefore, we have to recalculate the kernel
    if n_points%1 == 0:
        kernelParameter = gradient_ascend_for_kernel_parameters(X,y)

    K = kernel(X, X,kernelParameter)
    K = K +  sy*np.eye(n_points+1)
    L = np.linalg.cholesky(K)  

    Xtest = np.linspace(-span, span, n).reshape(-1,1) 
    # compute the mean at our test points.   
    Lk = np.linalg.solve(L, kernel(X, Xtest,kernelParameter))   
    mu = np.dot(Lk.T, np.linalg.solve(L, y))    

    K_ = kernel(Xtest, Xtest,kernelParameter)                   
    s2 = np.diag(K_) - np.sum(Lk**2, axis=0)    
    s = np.sqrt(s2).reshape(-1,1)
            
    # draw samples from the posterior at our test points after reaching the limit number of queries.
    pl.figure(point)
    pl.clf()
    pl.plot(X, y, 'r+', ms=20)
    pl.plot(Xtest, f(Xtest), 'b-')
    pl.gca().fill_between(Xtest.flat, (mu-3*s).flat, (mu+3*s).flat, color="#dddddd")
    pl.plot(Xtest, mu, 'r--', lw=2)
    pl.title('Mean predictions plus 3 st.deviations') 
    pl.axis([-span, span, -5, 5])
    pl.show()
    
# Next steps: check if it works for higher dimensional data, and use PSO instead of gradient ascend to optimize the parameter 
# （I assume PSO to be better, since gradient ascend is essentially a greedy algorithm on a non convex surface, and can easily stuck at local minima）.





