import numpy as np
import scipy as sc



A = np.random.rand(3,4)
b = np.random.rand(3,1)
m, n = A.shape

def opt_f(A,x,y,b):
    P = np.eye(len(x))
    F = x.T @ P @ x
    return F.ravel()

