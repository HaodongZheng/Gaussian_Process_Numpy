from __future__ import division
import numpy as np
import matplotlib.pyplot as pl

""" This is code for simple GP regression. It assumes a zero mean GP Prior """


# This is the true unknown function we are trying to approximate
f = lambda x: np.sin(0.9*x+ 0.4).flatten()
# f = lambda x: (0.25*(x**2) - 0.1*(x**3) + x).flatten()


# Define the kernel
def kernel(a, b,l=0.1):
    """ GP squared exponential kernel """
    # similar to rbf kernel
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
   # print(np.sum(a**2,1).reshape(-1,1).shape)
  #  print(np.sum(b**2,1).shape)
   # print(2*np.dot(a, b.T).shape)
  #  print(sqdist.shape)
    return np.exp(-.5 * (1/l) * sqdist)

def gradient_ascend_for_kernel_parameters(X,y,kernelParameter = 0.1):
    iterations = 2000
    K = kernel(X, X, kernelParameter)
    n = K.shape[0]
    K = K + 1e-6*np.eye(n)
    #XXT = np.dot(X,X.T)
    #print(f"XXT {XXT.shape} {XXT}")
    #XXR = np.diag(XXT).reshape((1,-1))
    #XXC = np.diag(XXT).reshape((-1,1))
    #print(f"XXR {XXR.shape} {XXR}")
    #print(f"XXC {XXC.shape} {XXC}")
    n = K.shape[0]
    #X_distance = (XXR + np.zeros((n,1))) + (XXC + np.zeros((1,n))) - 2* XXT
    step_length = 0.001
    sqdist = np.sum(X**2,1).reshape(-1,1) + np.sum(X**2,1) - 2*np.dot(X, X.T)

    for i in range(0,iterations):
        # here dKdl formula is correct but numerically unstable, better way to do it.
    #    dKdl = - K* np.log(K)/kernelParameter #checked
    #   dKdl = 0.5 * (K- 1e-6*np.eye(n))* X_distance/(kernelParameter**2)
        dKdl = 0.5 * (K- 1e-6*np.eye(n))* sqdist/kernelParameter**2 
    #    print(f"dKdl {dKdl.shape} {np.sum(dKdl)}")
        #print(K)
        L = np.linalg.cholesky(K) 
        L_inverse =  np.linalg.inv(L)
        K_inverse = np.dot(L_inverse.T,L_inverse)
       # print(f"K_inverse {K_inverse.shape}")
        alpha  = np.dot(K_inverse,y).reshape(-1,1)
       # print(f"alpha {alpha.shape}")
        A = np.dot(alpha,alpha.T) - K_inverse
       # print(f"A {A.shape}")
        B = np.dot(A,dKdl)
       # print(f"B {B.shape}")
        gradient = 0.5 * np.sum(np.diag(B),0)
       # print(f"gradient {gradient}")
        l = kernelParameter + step_length*gradient
        if l < 1e-2:
            l = 1e-2
        if l > 100:
            l = 100
        #K = (K-1e-6*np.eye(n))**(kernelParameter/l)+ 1e-6*np.eye(n)
        K = kernel(X, X, l) + 1e-6*np.eye(n)
       # print(kernel(X, X, l))
       # print(K)
       # print(f"K {K.shape}")
        kernelParameter = l

    print(f"iteration {i + 1}: kernelParameter = {kernelParameter}")
    return kernelParameter    

    

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
#kernelParameter = 0.0001
K = kernel(X, X,kernelParameter)
K = K + sy*np.eye(N)
# noicy gaussian process, variance s at each data point observation, they are independent on each other.
L = np.linalg.cholesky(K)  

# points we're going to make predictions at.
Xtest = np.linspace(-span, span, n).reshape(-1,1) 

# compute the mean at our test points.   
Lk = np.linalg.solve(L, kernel(X, Xtest,kernelParameter))   # mu = mu + kx*.T L^-T L^-1 y   K_y = Kxx = LL.T, here we first get L^-1 k* by solving x for Lx = kx*, x = L^-1 kx*, since L is a lower triangle one, this is relatively simple to do, by forward substitution. here mu is zero in the prior.
mu = np.dot(Lk.T, np.linalg.solve(L, y))    # here we calculate kx*.T L^-T, solve x for Lx = y, get L^-1 y, then take the dot product.

# compute the variance at our test points.
K_ = kernel(Xtest, Xtest,kernelParameter)                   # K**

s2 = np.diag(K_) - np.sum(Lk**2, axis=0)    # Lk = L^-1 kx*  NEW sigma = sigma** - sigma21 sigma11^-1 sigma12 = k** - kx*.T Kxx^-1 kx*  triangle matrix's inverse is also a triangle matrix, upper to upper, lower to lower
                                            # is L.T ^ -1 = L^-1 .T  ?   The answer interestingly is yes, consider  L^-1 L = I   and   L.T L.T^-1 = I (the multiplication order is important). therefore kx*.T Kxx^-1 kx* = Lk.T Lk, extract only diagonal, we need variance not covariance to show.
s = np.sqrt(s2)                             # this is the standard deviation.


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
pl.axis([-span, span, -20, 20])

# draw samples from the prior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
f_prior = np.dot(L, np.random.normal(size=(n,10)))
# this is due to the fact that every multivariate gaussian distribution can be written as X = AZ + mu, Z is a vector of n independent standard normal distribution, and A A.T = K, now L is a perfect A, so we can first sample from Z then do linear transformation to obtain X.
pl.figure(2)
pl.clf()
pl.plot(Xtest, f_prior)
pl.title('Ten samples from the GP prior')
pl.axis([-span, span, -20, 20])
pl.savefig('prior.png', bbox_inches='tight')

# draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,10)))
pl.figure(3)
pl.clf()
pl.plot(Xtest, f_post)
pl.title('Ten samples from the GP posterior')
pl.axis([-span, span, -20, 20])
pl.savefig('post.png', bbox_inches='tight')

pl.show()

pp = input("enter to terminate")
if pp == "":
    exit()

# find the best trade off point to query x_next
# Todo
# this seems to be ok, but in fact, i want to implement bayesian learning here. # thomson sampling, most probable improvement, etc. 
# and also tune the parameter of the kernel l and sigma y by maximizing the joint probability of the known points.
#define utility function to trade off exploration and exploitation  U = mu(x) + C sigma(x)
C = 1000
step = 0.01
limit = int(input("how many points to query:"))
L = np.linalg.cholesky(K)  
for point in range(0,limit):
    x_next = np.random.uniform(span, -span, size=(1,d))
    #K_inverse = np.linalg.inv(K)
    L_inverse =  np.linalg.inv(L)
    K_inverse = np.dot(L_inverse.T,L_inverse)
    for t in range(0,1000):
        X_diff = X - x_next
        #print(f"x_diff shape: {X_diff.shape} type {type(X_diff)}")
        k_x_star = kernel(X,x_next,kernelParameter)
        #K_inverse = np.linalg.inv(K)
        #print(f"k_x_star shape: {k_x_star.shape} type {type(k_x_star)}")
        n_points = X.shape[0]
        dummy =  np.zeros((n_points,n_points))
        np.fill_diagonal(dummy,k_x_star)
        J_dk_x_star_dx_T = np.dot(X_diff.T,dummy)
        #print(f"J_dk_x_star_dx_T shape: {J_dk_x_star_dx_T.shape} type {type(J_dk_x_star_dx_T)}")
        #print(f"y shape: {y.shape} type {type(y)}")
        #print(np.dot(J_dk_x_star_dx_T,K_inverse).shape)
        grad = np.dot(np.dot(J_dk_x_star_dx_T,K_inverse),y.reshape((-1,1))-2*C*k_x_star)
        #print(grad.shape)
        
        x_next = x_next + step*grad

    if x_next > span:
        x_next = span - np.random.uniform(0,span,size=(1,1))
    if x_next < -span:
        x_next = -span + np.random.uniform(0,span,size=(1,1))
    
    print(f"best next point {x_next}")
    y_next = f(np.array(x_next)) + sy*np.random.randn(1)
   # K_next = np.block([[K,k_x_star],[k_x_star.T, kernel(x_next, x_next,kernelParameter)+sy]])   
    X = np.block([[X],[x_next]])
    y = np.block([[y.reshape(-1,1)],[y_next]])  

    
    kernelParameter_last = kernelParameter
    if n_points%1 == 0:
        kernelParameter = gradient_ascend_for_kernel_parameters(X,y)
  #  K = K_next**(kernelParameter_last/kernelParameter)
    #kernelParameter = 0.1
    K = kernel(X, X,kernelParameter)
    K = K +  sy*np.eye(n_points+1)
    # print(kernelParameter)
    L = np.linalg.cholesky( K )  

    # points we're going to make predictions at.
    Xtest = np.linspace(-span, span, n).reshape(-1,1) 

    # compute the mean at our test points.   
    Lk = np.linalg.solve(L, kernel(X, Xtest,kernelParameter))   
    # mu = mu + kx*.T L^-T L^-1 y   K_y = Kxx = LL.T, here we first get L^-1 k* by solving x for Lx = kx*, x = L^-1 kx*.
    # since L is a lower triangle one, this is relatively simple to do, by forward substitution. here mu is zero in the prior.
    mu = np.dot(Lk.T, np.linalg.solve(L, y))    
    # here we calculate kx*.T L^-T, solve x for Lx = y, get L^-1 y, then take the dot product.

    # compute the variance at our test points.
    # K** is 
    K_ = kernel(Xtest, Xtest,kernelParameter)                   
    # Lk = L^-1 kx*  NEW sigma = sigma** - sigma21 sigma11^-1 sigma12 = k** - kx*.T Kxx^-1 kx*  triangle matrix's inverse is also a triangle matrix, upper to upper, lower to lower
    # is L.T ^ -1 = L^-1 .T  ?   The answer interestingly is yes, consider  L^-1 L = I   and   L.T L.T^-1 = I (the multiplication order is important). therefore kx*.T Kxx^-1 kx* = Lk.T Lk, extract only diagonal, we need variance not covariance to show.
    s2 = np.diag(K_) - np.sum(Lk**2, axis=0)    
    # this is the standard deviation.
    s = np.sqrt(s2).reshape(-1,1)                             


#plot pl.figure(1)
pl.figure(point)
pl.clf()
pl.plot(X, y, 'r+', ms=20)
pl.plot(Xtest, f(Xtest), 'b-')
pl.gca().fill_between(Xtest.flat, (mu-3*s).flat, (mu+3*s).flat, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.title('Mean predictions plus 3 st.deviations') 
pl.axis([-span, span, -5, 5])
pl.show()
    
#pp = input()
#if pp == "exit":
#    exit()
# Next steps: check if it works for higher dimensional data, and use PSO instead of gradient ascend for optimizing the parameter 
# （I assume this to be better, since gradient ascend is essentially a greedy algorithms in a non convex surface, and can easily stuck at local minima）.

