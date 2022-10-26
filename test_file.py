import jax.numpy as jnp
from jax import jit, vmap, grad, lax, random

key = random.PRNGKey(23)

x_data_1 = (jnp.linspace(0, 10, 200)).reshape((-1,1))
x_data_2 = random.shuffle(key=key,x=jnp.linspace(-3, 6, 200).reshape((-1,1)))
X_data = jnp.concatenate((x_data_1,x_data_2,jnp.ones((200,1))),axis=1)
std_ = 1.5
noise = random.normal(key, shape=(200,1)) * std_
theta = jnp.array([2,-6,4]).reshape((-1,1))

y = X_data@theta + noise

y
def model(par):
    """
    given an input of the data, the output of the model is returned
    :param par:
    :return:
    """
    return X_data@par

def log_posteriori_function(par: jnp.ndarray = None, estimations: jnp.ndarray = None):
    """
    The log of the posteriori distribution
    :param estimations:
    :param par: The matrix of the parmaeters
    :return:
    """

