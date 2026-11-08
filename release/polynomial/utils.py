import numpy as np
from scipy.io import *

# Read the mpg dataset
# t is a n-by-1 vector of target values, miles per gallon
# X is n-by-d matrix of input variables, each row is one training example
def loadData():
    data = np.loadtxt('auto-mpg.data', usecols=(0,1,2,3,4,5,6,7))
    t = data[:,0]
    X = data[:,1:]

    # randomize rows, there is structure in the ordering of the rows in auto-mpg.data.
    # use a fixed random permutation
    # if interested, see what happens with a real random permutation:
    #rp = np.random.permutation(X.shape[0])
    # note for compaitiblity with matlab, the stored rp may start from 1 instead of 0
    rp = np.squeeze(loadmat('rp.mat')['rp']-1)

    X = X[rp,:]
    t = t[rp]

    return t,X

# P = degexpand(X,k)
# X is n-by-d, n points in d-dim space
# Expand X to include powers from degree 0 to k (no cross-terms)
def degexpand(X,k):
    (n,d) = X.shape

    P = np.ones((n,1))
    for i in range(k):
        P = np.hstack((P,X**(i+1)))

    return P

# normalize each component of X to have mean 0 and convariance 1.
# X is N-by-D
# X_n is N-by-D
# N datapoints in D-dim space
def normalizeData(X):
    N = X.shape[0]
    mu = np.mean(X,axis=0)
    sig = np.std(X,axis=0)
    
    X = (X - mu) / sig # python broadcasting
    return X
    
