import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as o
from math import *


s=sqrt(0.75)

# Function to map an interior index (i,j) into the vector index
def ind(i,j,n):
    return i-1+(j-1)*(2*n-2-j)//2

# Source term for the desired reference solution
def fun(x,y):
    return -8*(sqrt(3)-2*y)*cos(y)+4*(-4-3*x+3*x*x+sqrt(3)*y-y*y)*sin(y)

# Reference solution
def uex(x,y):
    return ((2*y-sqrt(3))**2-3*(2*x-1)**2)*sin(y)

# Function to check whether an index pair (i,j) is the interior of the grid
def inrange(i,j,n):
    return i>0 and j>0 and i+j<n

def poisson_tri(n):
    h=1./n
    m=(n-1)*(n-2)//2

    # Create derivative matrix and source term
    d=np.zeros((m,m))
    f=np.empty((m))
    hfac=1/(3*h*h)
    for j in range(1,n):
        for i in range(1,n-j):
            ij=ind(i,j,n)

            # Derivative matrix
            d[ij,ij]=-12*hfac
            if(inrange(i+1,j,n)): d[ij,ind(i+1,j,n)]=2*hfac
            if(inrange(i,j-1,n)): d[ij,ind(i,j-1,n)]=2*hfac
            if(inrange(i-1,j+1,n)): d[ij,ind(i-1,j+1,n)]=2*hfac

            # Additional lines for the stencil in part (b)
            if(inrange(i-1,j,n)): d[ij,ind(i-1,j,n)]=2*hfac
            if(inrange(i,j+1,n)): d[ij,ind(i,j+1,n)]=2*hfac
            if(inrange(i+1,j-1,n)): d[ij,ind(i+1,j-1,n)]=2*hfac

            # Source term
            f[ij]=fun(h*(i+0.5*j),h*s*j)

    # Display the sparsity structure of the derivative matrix
    #plt.spy(d)
    #plt.show()

    # Solve the linear system
    u=np.linalg.solve(d,f)
    # Compute error
    uerr=0
    for j in range(1,n):
        for i in range(1,n-j):
            du=uex(h*(i+0.5*j),h*s*j)-u[ind(i,j,n)]
            uerr+=du*du
    return sqrt(s*uerr/(2*n*n))


def f(n,a,b):
    return a*np.power(n,b)
# Loop over a range of grid sizes and compute the error
def timer():
    times = []
    val_set = [10,20,40,80,160]
    for n in val_set:
        start = time.time()
        poisson_tri(n)
        end = time.time()
        print("Time for n = " + str(n) + ": " + str(end-start) + " seconds")
        times.append(end-start)
    params = np.polyfit(np.log(val_set), np.log(times), 1)
    plt.xscale("log")
    plt.yscale("log")
    plt.scatter(val_set,times)
    plt.plot(val_set, val_set**(params[0])*(8e-7), color='r')
    plt.show()

timer()