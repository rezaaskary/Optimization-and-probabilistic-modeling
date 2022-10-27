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

def model(par: jnp.ndarray = None) -> jnp.ndarray:
    """
    given an input of the data, the output of the model is returned. There is no need to parallelize the function or
     write in the vectorized format
    :param par: The array of model parameters given by the sampler (ndimx1)
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
    return 1

nchains = 25
theta_init = random.uniform(key=key, minval=0, maxval=1.0, shape=(len(theta), nchains))

nchains
# from sampler_algorithms import MetropolisHastings
# MetropolisHastings(log_prop_fcn=log_posteriori_function, model=model,
#                    iterations=150, chains=nchains,
#                    progress_bar=True, burnin=30, parallelized=True)