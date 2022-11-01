import jax.numpy as jnp
from jax import jit, vmap, grad, lax, random
from jnx_Probablity_Distribution_Continuous import Uniform, HalfNormal
from sampler_algorithms import MetropolisHastings, ModelParallelizer
import matplotlib.pyplot as plt


key = random.PRNGKey(23)
x_data_1 = (jnp.linspace(0, 10, 200)).reshape((-1, 1))
x_data_2 = random.shuffle(key=key, x=jnp.linspace(-3, 6, 200).reshape((-1, 1)))
# X_data = jnp.concatenate((x_data_1, x_data_2, jnp.ones((200, 1))), axis=1)
X_data = jnp.concatenate((x_data_1, jnp.ones((200, 1))), axis=1)

std_ = 0.5
noise = random.normal(key, shape=(200, 1)) * std_
theta = jnp.array([2, -6]).reshape((-1, 1))
y = X_data @ theta + noise
y = jnp.tile(y,(1,4))
theta2 = random.normal(key, shape=(2, 45))


def model(par: jnp.ndarray = None, X: jnp.ndarray = None) -> jnp.ndarray:
    """
    given an input of the data, the output of the model is returned. There is no need to parallelize the function or
    write in the vectorized format
    :param par: The array of model parameters given by the sampler (ndim x 1)
    :return: The model output (1 x 1)
    """
    return X @ par


def model_no_input(par: jnp.ndarray = None) -> jnp.ndarray:
    """
    given an input of the data, the output of the model is returned. There is no need to parallelize the function or
    write in the vectorized format
    :param par: the input parameters should be entered in the shape of (ndim x 1)
    :return: Thr output is returned to the shape of (Cx0)
    """
    return (par ** 2).sum()


D = ModelParallelizer(model=model, activate_jit=True, has_input=True)
# values = D.model_evaluate(theta2, X_data)
# values_der = D.diff_model_evaluate(theta2, X_data)
#
# D2 = ModelParallelizer(model=model_no_input, activate_jit=False, has_input=False, chains=45)
# values2 = D2.model_evaluate(theta2)
# values_der2 = D2.diff_model_evaluate(theta2)


from jnx_Probablity_Distribution_Continuous import Uniform

theta1 = Uniform(lower=-10, upper=10, activate_jit=True)
theta2 = Uniform(lower=-10, upper=10, activate_jit=True)
theta3 = Uniform(lower=-10, upper=10, activate_jit=True)
# sigma =  HalfNormal(sigma=4)


def liklihood(N, estimated: jnp.ndarray, measured: jnp.ndarray, sigma):
    error = ((estimated - measured) ** 2).sum(axis=1)
    return ((sigma * jnp.sqrt(2 * jnp.pi)) ** (-N)) * jnp.exp((-0.5 / sigma ** 2) * error)
def log_liklihood(N, estimated: jnp.ndarray, measured: jnp.ndarray, sigma) -> jnp.ndarray:
    return -N * jnp.log(sigma * jnp.sqrt(2 * jnp.pi)) - (0.5 / sigma ** 2) * ((estimated - measured) ** 2).sum(axis=0)


def log_posteriori_function(par: jnp.ndarray = None):
    """
    The log of the posteriori distribution
    :param par: The matrix of the parameters
    :return:
    """
    estimation = D.model_evaluate(par, X_data)
    log_par1 = theta1.log_pdf(par[0:1, :])
    log_par2 = theta2.log_pdf(par[1:2, :])
    # log_par3 = theta3.log_pdf(par[2:3, :])
    ll = log_liklihood(N=200, estimated=estimation, measured=y, sigma=0.5)
    return log_par2 + log_par1 + ll


nchains = 4
theta_init = random.uniform(key=key, minval=0, maxval=1.0, shape=(len(theta), nchains))

T = MetropolisHastings(log_prop_fcn=log_posteriori_function,
                       iterations=1000, chains=nchains, x_init=theta_init,
                       progress_bar=True, burnin=0, activate_jit=True, random_seed=65)
S1,S2 = T.sample()

plt.figure(dpi=100)
plt.plot(S1[0, 0, :])
plt.show()

plt.figure(dpi=100)
plt.plot(S1[1, 0, :])
plt.show()

# plt.figure(dpi=100)
# plt.plot(S1[2, 0, :])
# plt.show()

#
#
#
#
#
#
#
#
#
#
#
#
# key = random.PRNGKey(23)
# x_data_1 = (jnp.linspace(0, 10, 200)).reshape((-1, 1))
# x_data_2 = random.shuffle(key=key, x=jnp.linspace(-3, 6, 200).reshape((-1, 1)))
# X_data = jnp.concatenate((x_data_1, x_data_2, jnp.ones((200, 1))), axis=1)
# std_ = 1.5
# noise = random.normal(key, shape=(200, 1)) * std_
# theta = jnp.array([2, -6, 4]).reshape((-1, 1))
# y = X_data @ theta + noise
#
#
# def model(par: jnp.ndarray = None) -> jnp.ndarray:
#     """
#     given an input of the data, the output of the model is returned. There is no need to parallelize the function or
#      write in the vectorized format
#     :param par: The array of model parameters given by the sampler (ndimx1)
#     :return:
#     """
#     return X_data @ par
#
#
# theta1 = Uniform(lower=-10, upper=10)
# theta2 = Uniform(lower=-10, upper=10)
# theta3 = Uniform(lower=-10, upper=10)
#
#
# def log_posteriori_function(par: jnp.ndarray = None, estimations: jnp.ndarray = None):
#     """
#     The log of the posteriori distribution
#     :param estimations:
#     :param par: The matrix of the parmaeters
#     :return:
#     """
#     lg1 = theta1.log_pdf()
#     return 1
#
#
# nchains = 25
# theta_init = random.uniform(key=key, minval=0, maxval=1.0, shape=(len(theta), nchains))
#
# from sampler_algorithms import MetropolisHastings
#
# T = MetropolisHastings(log_prop_fcn=log_posteriori_function, model=model,
#                        iterations=150, chains=nchains,
#                        progress_bar=True, burnin=30, activate_jit=False)
