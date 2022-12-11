import numpy as np
import matplotlib.pyplot as plt
from jax import random, vmap, lax, jacfwd, jit, value_and_grad
import jax.numpy as jnp
import jax
import optax
from sklearn.decomposition import FactorAnalysis
import pandas as pd
from Ordinary_Differential_Equation_Solvers import ODESolvers


def ode_fcn(x: jnp.ndarray = None, p: jnp.ndarray = None, t: jnp.ndarray = None, u: jnp.ndarray = None) -> jnp.ndarray:
    m = 4  # the number of state variables
    dx_dt = jnp.zeros((m, 1))  # reallocating the values of state variables
    dx0_dt = -0.2 * x[0] + p[0] * x[1] * x[2] + 0.1 * u[2]
    dx1_dt = -0.1 * x[1] + p[1] * jnp.sin(x[1] + x[3]) + 0.3 * u[1]
    dx2_dt = -0.6 * x[2] + jnp.cos(x[2] + x[0]) - 0.2 * u[2] ** 2
    dx3_dt = -0.9 * x[3] + 0.01 * jnp.sin(x[3] + x[1]) * jnp.cos(x[2] + x[0]) - 0.2 * u[0]

    dx_dt = dx_dt.at[0].set(dx0_dt)
    dx_dt = dx_dt.at[1].set(dx1_dt)
    dx_dt = dx_dt.at[2].set(dx2_dt)
    dx_dt = dx_dt.at[3].set(dx3_dt)

    return dx_dt

n_par = 2
chains = 100
L = 500
par = jax.random.uniform(key=jax.random.PRNGKey(7), minval=-4, maxval=4, shape=(n_par, chains), dtype=jnp.float64)
x_0 = jax.random.uniform(key=jax.random.PRNGKey(7), minval=-4, maxval=4, shape=(4, chains), dtype=jnp.float64)
u = jax.random.uniform(key=jax.random.PRNGKey(7), minval=-4, maxval=4, shape=(3, L), dtype=jnp.float64)

















# from Probablity_distributions import *
# from tensorflow_probability.substrates.jax import distributions
#
# x = random.randint(key=random.PRNGKey(7), shape=(1000, 1), minval=0, maxval=10, dtype=int)
#
# M = distributions.Binomial(total_count=20, probs=0.5)
#
# RR = M.prob([0, 0.2, 1, 2, 3, 4, 5, 55])
#
# D = np.eye(5) * 6
# EE = np.power(D, -0.5)
#
# data = pd.read_csv('winequality-white.csv', delimiter=';')
# data = jnp.array(data.values[:, :-2])
