import jax.numpy as jnp
from jax import jit, vmap, grad, lax, random

key = random.PRNGKey(23)

x_data_1 = jnp.linspace(0, 10, 200)
x_data_2 = jnp.linspace(-3, 6, 200)
std_ = 3
noise = random.normal(key, shape=(200,)) * 3
y = 2 * x_data_1 - 6 * x_data_2 + noise


def model(par):
    """
    given an input of the data, the output of the model is returned
    :param par:
    :return:
    """
    return par[0] * x_data_1 + par[1] * x_data_2

def log_posteriori_function(par: jnp.ndarray = None, estimations: jnp.ndarray = None):
    """
    The log of the posteriori distribution
    :param estimations:
    :param par: The matrix of the parmaeters
    :return:
    """
    estimations = vmap(model, in_axes=)
