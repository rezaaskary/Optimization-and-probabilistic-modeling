import numpy as np
import matplotlib.pyplot as plt
from jax import random, vmap, lax, jacfwd, jit, value_and_grad
import jax.numpy as jnp
import optax
from sklearn.decomposition import FactorAnalysis
import pandas as pd

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



