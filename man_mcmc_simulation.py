import numpy as np
import matplotlib.pyplot as plt
from jax import random, vmap, lax, jacfwd, jit, value_and_grad
import jax.numpy as jnp
import jax
import optax
from sklearn.decomposition import FactorAnalysis
import pandas as pd
from Ordinary_Differential_Equation_Solvers import ODESolvers


# from Ordinary_Differential_Equation_Solvers as odes


def ode_fcn(x: jnp.ndarray = None, p: jnp.ndarray = None, t: int = None, u: jnp.ndarray = None) -> jnp.ndarray:
    dx_dt = jnp.zeros((4,))  # reallocating the values of state variables
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
L = 5000
# par = jax.random.uniform(key=jax.random.PRNGKey(7), minval=-4, maxval=4, shape=(n_par, chains), dtype=jnp.float64)
par = jnp.ones((n_par, chains))
u = jnp.zeros((3, L), dtype=jnp.float32)
u = u.at[0, 100:].set(0.4)
u = u.at[1, 200:].set(-1)
u = u.at[2, 300:].set(2)

x_0 = jax.random.uniform(key=jax.random.PRNGKey(7), minval=-4, maxval=4, shape=(4, chains), dtype=jnp.float64)
# u = jax.random.uniform(key=jax.random.PRNGKey(7), minval=-4, maxval=4, shape=(3, L), dtype=jnp.float64)
## passed
# euler


odes1 = ODESolvers(fcn=ode_fcn, steps=L, duration=50, n_sim=chains, n_input=3, n_states=4, n_params=3, x0=x_0,
                   method='euler', activate_jit=True)
# T1 = odes1.solve(parameter=par, u=u)
# plt.figure(dpi=150)
# plt.plot(T1[0, 0, :], '.')
# plt.plot(T1[1, 0, :], '.')
# plt.plot(T1[2, 0, :], '.')
# plt.plot(T1[3, 0, :], '.')
#
# odes2 = ODESolvers(fcn=ode_fcn, steps=L, duration=50, n_sim=chains, n_input=3, n_states=4, n_params=3, x0=x_0,
#                    method='ABAM5', activate_jit=True)
# T2 = odes2.solve(parameter=par, u=u)

# plt.plot(T2[0, 0, :], '-')
# plt.plot(T2[1, 0, :], '-')
# plt.plot(T2[2, 0, :], '-')
# plt.plot(T2[3, 0, :], '-')
# plt.show()





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
import SALib
from SALib.sample.fast_sampler import sample

problem = {
    'names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6' ],
    'num_vars': 6,
    'bounds': [[-jnp.pi, jnp.pi], [1.0, 0.2], [3, 0.5], [3, 0.5], [3, 0.5], [3, 0.5]],
    'groups': ['G1', 'G2', 'G1', 'G1', 'G1', 'G1'],
    'dists': ['unif', 'lognorm', 'triang', 'triang', 'triang', 'triang']
}

sample(problem=problem,N=2048,M=5,seed=3)