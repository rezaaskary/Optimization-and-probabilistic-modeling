import numpy as np
import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp
# from Probablity_distributions import *
from tensorflow_probability.substrates.jax import distributions


x  = random.randint(key=random.PRNGKey(7),shape=(1000, 1),minval=0,maxval=10,dtype=int)

M = distributions.Binomial(total_count=20,probs=0.5)

RR = M.prob([0,0.2,1,2,3,4,5,55])


D= np.eye(5)*6
EE = np.power(D,-0.5)
EE


class FactorAnalysis:
    def __init__(self,
    x: np.ndarray = None,
    n_comp: int = None,
    tolerance: float = 1e-8,
    max_iter: int = 1000,
    random_seed: int = 1,) -> None:
        if isinstance(x, np.ndarray):
            if np.any(np.isnan(x)):
                raise Exception(f'There are NaN values in the input matrix!')
            else:
                self.x = x
                self.n, self.p = self.x.shape
                self.mean = self.x.mean(axis=0)
                self.var  = self.x.var(axis=0)
                self.x_m = (self.x - np.tile(self.mean,reps=(self.n,1))).T
        else:
            raise Exception(f'The format of {type(x)} is not supported.\n'
             f'The input matrix should be given in ndarray format.')
        
        if isinstance(n_comp, int):
            if n_comp < 1:
                raise Exception('The minimum number of principal components should be a positive integer.')
            else:
                self.n_comp = n_comp
        elif not n_comp:
            self.n_comp = self.p
        else:
            raise Exception('The format of the number of component is not supported.\n'
            ' Please enter the number of components as a positive integer!')
        
        if isinstance(tolerance, float):
            if tolerance > 1:
                raise Exception('Please enter a small value for tolerance. Ex. 1e-6')
            else:
                self.tolerance = tolerance
        elif not tolerance:
            self.tolerance = 1e-8
        else:
            raise Exception('The format of tolerance is not supported.\n'
            ' Please enter a small value as tolerance (Ex. 1e-8)')

        if isinstance(max_iter, int):
            if max_iter < 1:
                raise Exception('Please enter a positive integer as the maximum number of iterations.')
            else:
                self.max_iter = max_iter
        elif not max_iter:
            self.max_iter = 1000
        else:
            raise Exception('The format of maximum iterations is not supported.\n'
            ' Please enter positive integer as maximum number of iterations (Ex. 1000)')
        self.psi = np.diag(np.power(np.ones((self.p,)), -0.5))
        
    
    def calculate(self):

        for i in range(self.max_iter):
            self.x_hat = self.psi @ self.x_m / np.sqrt(self.n)
            u,s,wh = np.linalg.svd(self.x_hat,
                        full_matrices=False,
                         compute_uv=True,
                          hermitian=False)
            a = s ** 2
            ah = a[:self.n_comp]
            uh = u[:, :self.n_comp]
            f = np.power(self.psi, 0.5)@uh@np.power(np.maximum(ah-1.0, np.finfo(float).eps), 0.5)[:,np.newaxis]
            liklihood = -0.5*self.n * (np.log(s[:self.n_comp]).sum() +
             self.n_comp + (s[self.n_comp:]).sum() + np.log(np.diag(self.psi).prod()*2*np.pi))
            print(liklihood)
            self.psi = np.diag(np.maximum(self.var - f[:,0]**2, np.finfo(float).eps))
        return




data = random.gamma(key=random.PRNGKey(23), a=0.2, shape=(5000, 5)).T
# data = np.array(data.T)

# D = FactorAnalysis(x=data,n_comp=2,max_iter=100).calculate()
data = data - jnp.tile(data.mean(axis=1),5000)


data
# def(data, psi, f):
#     return