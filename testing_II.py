import pandas as pd
import numpy as np
import sklearn
from sklearn import decomposition

import jax.numpy as jnp
from jax import lax
import jax
from functools import partial

idx_new = jnp.arange(start=1, stop=10, dtype=jnp.int32)
idex_old = jnp.arange(start=0, stop=9, dtype=jnp.int32)

def dummy_df(i: int, values: tuple) -> tuple:
    idex_old, idx_new = values
    slice = lax.dynamic_slice(operand=idex_old, start_indices=(0,), slice_sizes=(i,))
    idx_new = lax.dynamic_update_slice(operand=idx_new, update=slice, start_indices=(0,))
    return idex_old, idx_new


lax.fori_loop(lower=0, upper=9, body_fun=dummy_df, init_val=(idex_old, idx_new))

data = pd.read_csv('winequality-white.csv', delimiter=';')
data = np.array(data.values[:, :-2])

# T = FactorAnalysis_(x=data, n_comp=2, tolerance=1e-6, max_iter=1000, random_seed=1)
# T.calculate()

# data = ((data - data.mean(axis=0))/data.std(axis=0))

pp = decomposition.FactorAnalysis(n_components=2, max_iter=10000, tol=1e-3, svd_method='lapack')
DD1 = pp.fit(data)
DD = pp.transform(data)
# ee =DD.get_covariance()
ew = DD1.components_.T @ DD1.components_ + np.diag(DD1.noise_variance_)
new_data = (data - data.mean(axis=0)).T
f = DD1.components_.T
CC = np.linalg.inv(f.T @ f + np.diag(DD1.noise_variance_)) @ f.T @ new_data

CC
