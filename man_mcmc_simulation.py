import numpy as np
from jax import random, lax
import jax.numpy as jnp

# from Probablity_distributions import *
from tensorflow_probability.substrates.jax import distributions

x = random.randint(key=random.PRNGKey(7), shape=(1000, 1), minval=0, maxval=10, dtype=int)

M = distributions.Binomial(total_count=20, probs=0.5)

RR = jnp.array([0, 0.2, 1, 2, 3, 4, 5, 55])

TT =RR < 1
















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
