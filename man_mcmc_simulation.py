import numpy as np
import matplotlib.pyplot as plt
from jax import random, vmap, lax, jacfwd, jit, value_and_grad
import jax.numpy as jnp
import optax
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
                 x: jnp.ndarray = None,
                 n_comp: int = None,
                 tolerance: float = 1e-8,
                 max_iter: int = 1000,
                 random_seed: int = 1,
                 method: str = 'EM') -> None:

        if isinstance(method, str) and method in ['sgd', 'EM']:
            self.method = method
        elif not method:
            raise Exception('Please enter the method of calculating the latent variables.')
        else:
            raise Exception('Please select from either svd or EM as the supported method for calculating the latent '
                            'variables.')

        if isinstance(random_seed, int):
            self.key = random.PRNGKey(random_seed)
        elif not random_seed:
            self.key = random.PRNGKey(1)
        else:
            raise Exception('Enter an integer as the value of seed for generating pseudo random numbers.')

        if isinstance(x, jnp.ndarray):
            if jnp.any(jnp.isnan(x)):
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
        self.eps = jnp.finfo(float).eps
        self.itr = 0.0
        self.psi = random.uniform(key=self.key,
                                  shape=(self.p,),
                                  minval=0,
                                  maxval=1)

        self.f = random.uniform(key=self.key,
                                shape=(self.p, self.n_comp),
                                minval=0,
                                maxval=1)

        def _cond_fun(values: tuple = None) -> bool:
            itr, _, _, _, likelihood_error = values
            error = jnp.abs(likelihood_error).astype(float)
            return (error > self.tolerance) | (itr > self.max_iter)

        def _em_factor_analysis(values: tuple = None) -> tuple:
            itr, psi, f, old_log_likelihood, log_likelihood_error = values
            x_hat = jnp.diag(psi ** -0.5) @ self.x_m / jnp.sqrt(self.n)
            u_svd, s_svd, _ = jnp.linalg.svd(x_hat, full_matrices=False)
            a_svd = s_svd ** 2
            f = jnp.diag(psi ** 0.5) @ u_svd[:, :self.n_comp] @ jnp.diag(
                jnp.maximum(a_svd[:self.n_comp] - 1.0, self.eps) ** 0.5)
            likelihood = -0.5 * self.n * (jnp.log(a_svd[:self.n_comp]).sum() +
                                          self.n_comp + (a_svd[self.n_comp:]).sum() + jnp.log(
                        jnp.linalg.det(jnp.diag(psi * 2 * jnp.pi))))
            psi = self.var - jnp.diag(f @ f.T)
            log_likelihood_error = likelihood - old_log_likelihood
            itr += 1
            return itr, psi, f, likelihood, log_likelihood_error

        if self.method == 'EM':
            self.body_fun = _em_factor_analysis
            self.cond_fun = _cond_fun

    def fit(self):

        self.itr, \
            self.psi, self.f, \
            self.log_likelihood, \
            self.log_likelihood_error = lax.while_loop(body_fun=self.body_fun, cond_fun=self.cond_fun,
                                                   init_val=(self.itr,
                                                             self.psi,
                                                             self.f,
                                                             jnp.array(-jnp.inf, dtype=jnp.float32),
                                                             jnp.array(-jnp.inf, dtype=jnp.float32)))
        self.covariance = self.f@self.f.T + jnp.diag(self.psi)

        return self

    def fit_transform(self):
        self.fit()
        coef = self.f/self.psi[:, jnp.newaxis]
        self.latent_variables = jnp.linalg.inv(coef.T@self.f + jnp.eye(self.n_comp))@coef.T@self.x_m
        return self




# data = random.gamma(key=random.PRNGKey(23), a=0.2, shape=(5000, 5)).T
# data = np.array(data.T)
#
# D = FactorAnalysis(x=data,n_comp=2,max_iter=100).calculate()
# data = data - jnp.tile(data.mean(axis=1)[:,jnp.newaxis], 5000)


data = pd.read_csv('winequality-white.csv', delimiter=';')
data = jnp.array(data.values[:, :-2])

T = FactorAnalysis_(x=data, n_comp=2, tolerance=1e-8, max_iter=500, random_seed=1)
T.fit_transform()

data2 = ((data - data.mean(axis=0)) / data.std(axis=0)).T
L = data2.shape[1]
f = random.uniform(key=random.PRNGKey(3), shape=(data2.shape[0], 2), minval=0.1, maxval=1)
psi = random.uniform(key=random.PRNGKey(3), shape=(data2.shape[0],), minval=0.1, maxval=1)
# actorAnalysis_(x=data, n_comp=2, tolerance=1e-6, max_iter=1000, random_seed=1)
# T.calc
import optax
import functools


# @functools.partial(vmap,in_axes=[1, None])
def fcn2(obs, invmat):
    return obs.T @ invmat @ obs


vfcn2 = vmap(fun=fcn2, in_axes=[1, None])


# f

def fcn1(data2, psi, f):
    sigma = f @ f.T + jnp.diag(psi)
    sig_inv = jnp.linalg.inv(sigma)
    out = -0.5 * vfcn2(data2, sig_inv).sum() - 0.5 * L * lax.log(jnp.linalg.det(2 * jnp.pi * sigma))
    return out


# grad1 = jacfwd(fun=fcn1, argnums=1)
# grad2 = jacfwd(fun=fcn1, argnums=2)
grad3 = jit(value_and_grad(fun=fcn1, argnums=[1, 2]))

# sd = grad1(data,jnp.ones((5,)), jnp.ones((5,2)))
# sd2 = grad2(data,jnp.ones((5,)), jnp.ones((5,2)))
# sd3 = grad3(data,jnp.ones((5,)), jnp.ones((5,2)))
# T = fcn1(data,jnp.ones((5,)), jnp.ones((5,2)))

# f = 0.1*jnp.ones((5,2))
# psi = 0.01*jnp.ones((5,))


# lr = jnp.arange()
lr = jnp.linspace(start=1e-4, stop=0.1, num=5000)
lr = 1e-6
# optimizer = optax.adam(0.02)
# Obtain the `opt_state` that contains statistics for the optimizer.
# params = {'f': f,'psi':psi}
# opt_state = optimizer.init(params)

#
# compute_loss = lambda params, x, y: optax.l2_loss(params['w'].dot(x), y)
# grads = jax.grad(compute_loss)(params, xs, ys)

# LL = jnp.inf
# for i in range(30000):
#     TT = (grad3(data2, psi, f))
#     # psip, fp = (grad3(data, psi, f))
#     # psip = grad1(data[:,i*50:(i+1)*50],psi, f)
#     # fp = grad2(data[:,i*50:(i+1)*50],psi, f)
#     fp = TT[1][1]
#     psip = TT[1][0]
#     f = f + lr * fp
#     psi = psi + lr * psip
#     # lik = fcn1(data, psi, f)
#     print(LL - TT[0])
#     LL = TT[0]
# f
# CC = f* jnp.tile( data.std(axis=0)[:,jnp.newaxis],reps=(1,2))
