import numpy as np
import matplotlib.pyplot as plt
from jax import random, vmap, lax, jacfwd, jit,value_and_grad
import jax.numpy as jnp
from sklearn.decomposition import FactorAnalysis
import pandas as pd


# from Probablity_distributions import *
from tensorflow_probability.substrates.jax import distributions

x = random.randint(key=random.PRNGKey(7), shape=(1000, 1), minval=0, maxval=10, dtype=int)

M = distributions.Binomial(total_count=20, probs=0.5)

RR = M.prob([0, 0.2, 1, 2, 3, 4, 5, 55])

D = np.eye(5) * 6
EE = np.power(D, -0.5)
EE

import numpy as np
class FactorAnalysis_:
    def __init__(self,
                 x: np.ndarray = None,
                 n_comp: int = None,
                 tolerance: float = 1e-8,
                 max_iter: int = 1000,
                 random_seed: int = 1, ) -> None:
        if isinstance(x, np.ndarray):
            if np.any(np.isnan(x)):
                raise Exception(f'There are NaN values in the input matrix!')
            else:
                self.x = x
                self.n, self.p = self.x.shape
                self.mean = self.x.mean(axis=0)
                self.var = self.x.var(axis=0)
                self.x_m = (self.x - np.tile(self.mean, reps=(self.n, 1))).T
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
            u, s, wh = np.linalg.svd(self.x_hat,
                                     full_matrices=False,
                                     compute_uv=True,
                                     hermitian=False)
            a = s ** 2
            ah = a[:self.n_comp]
            uh = u[:, :self.n_comp]
            f = np.power(self.psi, 0.5) @ uh @ np.power(np.maximum(ah - 1.0, np.finfo(float).eps), 0.5)[:, np.newaxis]
            liklihood = -0.5 * self.n * (np.log(s[:self.n_comp]).sum() +
                                         self.n_comp + (s[self.n_comp:]).sum() + np.log(
                        np.diag(self.psi).prod() * 2 * np.pi))
            print(liklihood)
            self.psi = np.diag(np.maximum(self.var - f[:, 0] ** 2, np.finfo(float).eps))
        return


# data = random.gamma(key=random.PRNGKey(23), a=0.2, shape=(5000, 5)).T
# data = np.array(data.T)
#
# D = FactorAnalysis(x=data,n_comp=2,max_iter=100).calculate()
# data = data - jnp.tile(data.mean(axis=1)[:,jnp.newaxis], 5000)


data = pd.read_csv('winequality-white.csv',delimiter=';')

data = jnp.array(data.values[:,:-2])
data = (data - data.mean(axis=0)).T
L = data.shape[0]
f = random.uniform(key=random.PRNGKey(3),shape=(data.shape[0], 2),minval=0.1,maxval=1)
psi = random.uniform(key=random.PRNGKey(3),shape=(data.shape[0], ),minval=0.1,maxval=1)

import optax
import functools


# @functools.partial(vmap,in_axes=[1, None])
def fcn2(obs, invmat):
    return obs.T @ invmat @ obs


vfcn2 = vmap(fun=fcn2, in_axes=[1, None])


# f

def fcn1(data, psi, f):
    sigma = f @ f.T + jnp.diag(psi)
    sig_inv = jnp.linalg.inv(sigma)
    out = -0.5 * vfcn2(data, sig_inv).sum() -0.5 * L * lax.log(jnp.linalg.det(2*jnp.pi*sigma))
    return out



grad1 = jacfwd(fun=fcn1,argnums=1)
grad2 = jacfwd(fun=fcn1,argnums=2)
grad3 = jit(value_and_grad(fun=fcn1,argnums=[1,2]))

# sd = grad1(data,jnp.ones((5,)), jnp.ones((5,2)))
# sd2 = grad2(data,jnp.ones((5,)), jnp.ones((5,2)))
# sd3 = grad3(data,jnp.ones((5,)), jnp.ones((5,2)))
# T = fcn1(data,jnp.ones((5,)), jnp.ones((5,2)))

# f = 0.1*jnp.ones((5,2))
# psi = 0.01*jnp.ones((5,))


# lr = jnp.arange()
lr = jnp.linspace(start=1e-4,stop=0.1,num=2000)


# optimizer = optax.adam(0.02)
# Obtain the `opt_state` that contains statistics for the optimizer.
# params = {'f': f,'psi':psi}
# opt_state = optimizer.init(params)

#
# compute_loss = lambda params, x, y: optax.l2_loss(params['w'].dot(x), y)
# grads = jax.grad(compute_loss)(params, xs, ys)




for i in range(2000):
    TT = (grad3(data, psi, f))
    psip, fp = (grad3(data,psi,f))
    # psip = grad1(data[:,i*50:(i+1)*50],psi, f)
    # fp = grad2(data[:,i*50:(i+1)*50],psi, f)

    f = f + lr[i] * fp
    psi = psi + lr[i] * psip
    lik = fcn1(data, psi, f)
    print(lik)






