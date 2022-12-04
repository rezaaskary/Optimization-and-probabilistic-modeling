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

data = pd.read_csv('winequality-white.csv', delimiter=';')
data = jnp.array(data.values[:, :-2])


class CanonicalCorrelationAnalysis:
    def __init__(self,
                 x: jnp.ndarray = None,
                 y: jnp.ndarray = None,
                 n_comp: int = None,
                 tolerance: float = 1e-8,
                 max_iter: int = 1000,
                 random_seed: int = 1,
                 method: str = 'EM') -> None:
        """

        :param x:
        :param y:
        :param n_comp:
        :param tolerance:
        :param max_iter:
        :param random_seed:
        :param method:
        """
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

        if isinstance(y, jnp.ndarray):
            if jnp.any(jnp.isnan(y)):
                raise Exception(f'There are NaN values in the input matrix y!')
            else:
                self.y = y
                self.n_y, self.p_y = self.y.shape
        else:
            raise Exception(f'The format of {type(y)} is not supported.\n'
                            f'The input matrix should be given in ndarray format.')

        if isinstance(x, jnp.ndarray):
            if jnp.any(jnp.isnan(x)):
                raise Exception(f'There are NaN values in the input matrix!')
            else:
                self.x = x
                self.n_x, self.p_x = self.x.shape
        else:
            raise Exception(f'The format of {type(x)} is not supported.\n'
                            f'The input matrix should be given in ndarray format.')

        if self.n_y == self.n_x:
            self.z = jnp.concatenate(arrays=(self.x, self.y), axis=1)
            self.mean = self.z.mean(axis=0)
            self.var = self.z.var(axis=0)
            self.z_m = (self.z - jnp.tile(self.mean, reps=(self.n_x, 1))).T

        else:
            raise Exception('Matrices x and y have different observations. They are not consistent.')



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
        self.covariance = self.f @ self.f.T + jnp.diag(self.psi)

        return self

    def fit_transform(self):
        self.fit()
        coef = self.f / self.psi[:, jnp.newaxis]
        return jnp.linalg.inv(coef.T @ self.f + jnp.eye(self.n_comp)) @ coef.T @ self.x_m
