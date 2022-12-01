import numpy as np
from jax import random, lax, vmap
import jax.numpy as jnp
import sklearn








data = random.gamma(key=random.PRNGKey(23), a=0.2, shape=(50, 5))
data = data.at[4, 2].set(jnp.nan)
data = data.at[5, 1].set(jnp.nan)
data = data.at[5, 2].set(jnp.nan)

ind = jnp.arange(0,50,dtype=int)
def df(i, value):
    # df = jnp.where(~jnp.isnan(value[i, :]))
    df = jnp.where(value[i, :]>0)
    val_sample = df

    return
def df2(value,row, colum, ind,miss):
    rr = row[5*ind:5*(ind+1)-1]
    # cc = colum[]
    # T = value[]
    rr
    return rr


f = jnp.where(~jnp.isnan(data))
f2 =  (jnp.isnan(data)).sum(axis=1)
miss = f2.cumsum()
T1 = (~jnp.isnan(data)).sum(axis=1)
rows = f[0]
column= f[1]

TT = data[rows[:5], column[rows[:5]]]


T = vmap(fun=df2,in_axes=[None,None,None,0,0], out_axes=0)(data, rows, column, ind,miss)
T
#
# lax.fori_loop(lower=0, upper=data.shape[0], body_fun=df, init_val=data)
#
# numpy.logical_not(numpy.isnan(x))












# from Probablity_distributions import *
from tensorflow_probability.substrates.jax import distributions

x = random.randint(key=random.PRNGKey(7), shape=(1000, 1), minval=0, maxval=10, dtype=int)

M = distributions.Binomial(total_count=20, probs=0.5)

RR = jnp.array([0, 0.2, 1, 2, 3, 4, 5, 55])

TT = RR < 1


def vc(x):
    if isinstance(x, list):
        return np.array(x)
    else:
        raise Exception('The format of the input variable is not supported!')


def log_posteriori_fcn():
    parameter1 = Uniform(a=0, b=4, return_der_pdf=False, return_der_logpdf=False)
    parameter2 = Normal(sigma=3, mu=0, return_der_pdf=False, return_der_logpdf=False)
    CC = parameter1.pdf(np.array(.1))

    return


T = log_posteriori_fcn()
