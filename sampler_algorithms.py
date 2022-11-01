import jax.numpy as jnp
from jax import vmap, jit, grad, random, lax
import scipy as sc
from tqdm import tqdm
import matplotlib.pyplot as plt


class ModelParallelizer:
    def __init__(self, model: callable = None, has_input: bool = True, chains: int = None, n_obs: int = None,
                 activate_jit: bool = False):
        """
        Parallelling the model function for the fast evaluation of the model as well as the derivatives of the model
        with respect to the model parameters. The model can be either in the format of y=f(theta,x) or y=f(theta).
        Constraint:  The model should  be a multi-input-single-output model.
        :param: chains: an integer indicating the number of chains used for parallel evaluation of the model
        :param model: Given an input of the data, the output of the model is returned. The model inputs are parameters
         (ndim x 1) and model input variables (N x s). For parallel evaluation, the model input would be (ndim x C).
         :param: n_obs: an integer indicating the number of observations (measurements) of the model
        :param activate_jit: A boolean variable used to activate(deactivate) just-in-time evaluation of the model
        """

        if isinstance(chains, int):
            self.chains = chains
        elif not chains:
            self.chains = None
        else:
            raise Exception('The number of chains (optional) is not specified correctly!')

        if isinstance(n_obs, int):
            self.n_obs = n_obs
        elif not n_obs:
            self.n_obs = None
        else:
            raise Exception('The number of observations (optional) is not specified correctly!')

        if isinstance(activate_jit, bool):
            self.activate_jit = activate_jit
        else:
            self.activate_jit = False
            print(
                f'---------------------------------------------------------------------------------------------------\n'
                f'The default value of {self.activate_jit} is selected for parallelized simulations\n'
                f'----------------------------------------------------------------------------------------------------')

        if isinstance(has_input, bool):
            self.has_input = has_input
            if self.has_input:
                print(f'---------------------------------------------------------\n'
                      ' you  have specified that your model is like y = f(theta,x)\n'
                      '----------------------------------------------------------')
            else:
                print(f'---------------------------------------------------------\n'
                      ' you  have specified that your model is like y = f(theta)\n'
                      ' ----------------------------------------------------------')
        else:
            raise Exception('Please specify  whether the model has any input other than model parameters')

        # parallelize the model evaluation as well as calculating the
        if hasattr(model, "__call__"):
            self.model_eval = model
            if self.has_input:
                model_val = vmap(vmap(self.model_eval,
                                      in_axes=[None, 0],  # this means that we loop over input observations(1 -> N)
                                      axis_size=self.n_obs,  # specifying the number of measurements
                                      out_axes=0),  # means that we stack the observations in rows
                                 in_axes=[1, None],  # means that we loop over chains (1 -> C)
                                 axis_size=self.chains,  # specifying the number of chains
                                 out_axes=1)  # means that we stack chains in columns
                model_der = vmap(vmap(grad(self.model_eval,
                                           argnums=0),  # parameter 0 means model parameters (d/d theta)
                                      in_axes=[1, None],  # [1, None] means that we loop over chains (1 -> C)
                                      out_axes=1),  # means that chains are stacked in the second dimension
                                 in_axes=[None, 0],  # [None, 0] looping over model inputs (1 -> N)
                                 axis_size=self.n_obs,  # the size of observations
                                 out_axes=2)  # staking
            else:
                def reshaped_model(inputs):
                    return self.model_eval(inputs)[jnp.newaxis]

                model_val = vmap(reshaped_model, in_axes=1, axis_size=self.chains, out_axes=1)
                model_der = vmap(grad(self.model_eval, argnums=0), in_axes=1, axis_size=self.chains, out_axes=1)
        else:
            raise Exception('The function of the model is not defined properly!')

        if self.activate_jit:
            self.model_evaluate = jit(model_val)
            self.diff_model_evaluate = jit(model_der)
        else:
            self.model_evaluate = model_val
            self.diff_model_evaluate = model_der

    @property
    def info(self):
        if self.has_input:
            print('----------------------------------------------------------\n'
                  'the input of the model should be in the format of:         \n'
                  'theta: (ndim x C). Where ndim indicate the dimension of the\n'
                  'problem. C also account for the number of chains (parallel \n'
                  'evaluation).\n'
                  'X(N x s): A matrix of the input (other than the model\n'
                  'parameters). N indicate the number of observations and\n'
                  's indicate the number of input variables.\n'
                  'Output:\n'
                  'y (N x C): parallelized valuation of the model output\n'
                  'dy/dt (ndim x C x N): the derivatives of the  model\n'
                  ' output with respect to each model parameters\n'
                  '----------------------------------------------------------')
        else:
            print('----------------------------------------------------------\n'
                  'the input of the model should be in the format of:         \n'
                  'theta: (ndim x C). Where ndim indicate the dimension of the\n'
                  'problem. C also account for the number of chains (parallel \n'
                  'evaluation).\n'
                  'Output:\n'
                  'y (1 x C): parallelized valuation of the model output\n'
                  'dy/dt (ndim x C): the derivatives of the  model\n'
                  ' output with respect to each model parameters\n'
                  '----------------------------------------------------------')
        return

        # if self.activate_jit:
        #     self.model_evaluate = jit(vmap(self.model_eval, in_axes=[None, 0], out_axes=0))
        #     self.diff_model_evaluate = jit(vmap(vmap(grad(self.model_eval,
        #                                                   argnums=0),  # parameter 0 means model parameters
        #                                              in_axes=[1, None],  # [1, None] means that we loop over chains
        #                                              out_axes=1),
        #                                         # means that chains are stacked in the second dimension
        #                                         in_axes=[None, 0],  # [None, 0] looping over model inputs
        #                                         out_axes=2))  # staking
        # else:
        #     self.model_evaluate = vmap(self.model_eval, in_axes=[None, 0], out_axes=0)
        #     self.diff_model_evaluate = vmap(vmap(grad(self.model_eval,
        #                                               argnums=0),  # parameter 0 means model parameters
        #                                          in_axes=[1, None],  # [1, None] means that we loop over chains
        #                                          out_axes=1),
        #                                     # means that chains are stacked in the second dimension
        #                                     in_axes=[None, 0],  # [None, 0] looping over model inputs
        #                                     out_axes=2)  # staking

    # def model_evaluate(self):
    #     """
    #     Parallelized model
    #     :return: The callable function for evaluating the model
    #     """
    #     return self.mdl_eval
    #
    # def diff_model_evaluate(self):
    #     """
    #     Parallelized derivatives of the model
    #     :return: The callable function for evaluating the model
    #     """
    #     return self.mdl_der_eval


class MetropolisHastings:
    def __init__(self, log_prop_fcn: callable = None, iterations: int = None, burnin: int = None,
                 x_init: jnp.ndarray = None, activate_jit: bool = False, chains: int = 1, progress_bar: bool = True,
                 random_seed: int = 1):
        """
        Metropolis Hastings sampling algorithm
        :param log_prop_fcn: Takes the log posteriori function
        :param iterations: The number of iteration
        :param burnin: The number of initial samples to be droped sowing to non-stationary behaviour
        :param x_init: The initialized value of parameters
        :param parallelized: A boolean variable used to activate or deactivate the parallelized calculation
        :param chains: the number of chains used for simulation
        :param progress_bar: A boolean variable used to activate or deactivate the progress bar
        :param model: The model function (a function that input parameters and returns estimations)
        """
        self.key = random.PRNGKey(random_seed)

        # checking the correctness of log probability function
        if hasattr(log_prop_fcn, "__call__"):
            self.log_prop_fcn = log_prop_fcn
        else:
            raise Exception('The log probability function is not defined properly!')

        # checking the correctness of the iteration
        if isinstance(iterations, int):
            self.iterations = iterations
        else:
            self.iterations = 1000
            print(f'-------------------------------------------------------------------------------------------------\n'
                  f'The iteration is not an integer value.\n'
                  f' The default value of {self.iterations} is selected as the number of iterations\n'
                  f'--------------------------------------------------------------------------------------------------')

        if isinstance(burnin, int):
            self.burnin = burnin
        elif burnin is None:
            self.burnin = 0
            print(f'-------------------------------------------------------------------------------------------------\n'
                  f'The number samples from dropping after simulation is not an integer value.\n'
                  f' The default value of {self.burnin} is selected as the number of burnin samples\n'
                  f'--------------------------------------------------------------------------------------------------')
        else:
            self.burnin = 0
            print(f'-------------------------------------------------------------------------------------------------\n'
                  f'The number samples from dropping after simulation is not an integer value.\n'
                  f' The default value of {self.burnin} is selected as the number of burnin samples\n'
                  f'--------------------------------------------------------------------------------------------------')

        if self.burnin >= self.iterations:
            raise Exception('The number of samples selected for burnin cannot be greater than the simulation samples!')

        # checking the correctness of the iteration
        if isinstance(chains, int):
            self.n_chains = chains
        else:
            self.n_chains = 1
            print(
                f'---------------------------------------------------------------------------------------------------\n'
                f'The number of chains is not an integer value.\n'
                f' The default value of {self.n_chains} is selected as the number of chains\n'
                f'----------------------------------------------------------------------------------------------------')

            # checking the correctness of initial condition
        if isinstance(x_init, jnp.ndarray):
            dim1, dim2 = x_init.shape
            if dim2 != self.n_chains:
                raise Exception('The initial condition is not consistent with the number of chains!')
            else:
                self.ndim = dim1
                self.x_init = x_init
        else:
            raise Exception('The initial condition is not selected properly!')

        # checking the correctness of the vectorized simulation
        if isinstance(activate_jit, bool):
            self.activate_jit = activate_jit
        else:
            self.activate_jit = False
            print(
                f'---------------------------------------------------------------------------------------------------\n'
                f'The default value of {self.activate_jit} is selected for parallelized simulations\n'
                f'----------------------------------------------------------------------------------------------------')

        # checking the correctness of the progressbar
        if isinstance(progress_bar, bool):
            self.progress_bar = not progress_bar
        else:
            self.progress_bar = False
            print(
                f'---------------------------------------------------------------------------------------------------\n'
                f'The progress bar is activated by default since the it is not entered by the user\n'
                f'----------------------------------------------------------------------------------------------------')

        # initializing chain values
        self.chains = jnp.zeros((self.ndim, self.n_chains, self.iterations))
        # initializing the log of the posteriori values
        self.log_prop_values = jnp.zeros((self.iterations, self.n_chains))
        # initializing the track of hasting ratio values
        self.accept_rate = jnp.zeros((self.iterations, self.n_chains))

        # initializing the first values of the log probability
        # self.log_prop_values[:, 0] = self.log_prop_fcn(self.x_init)

        # in order to calculate the acceptance ration of all chains
        self.n_of_accept = jnp.zeros((1, self.n_chains))

    def sample(self):
        """
        vectorized metropolis-hastings sampling algorithm used for sampling from the posteriori distribution
        :returns: chains: The chains of samples drawn from the posteriori distribution
                  acceptance rate: The acceptance rate of the samples drawn form the posteriori distributions
        """
        sigma = 0.1
        rndw_samples = random.normal(key=self.key, shape=(self.ndim, self.n_chains, self.iterations)) * sigma
        self.chains = self.chains.at[:, :, 0].set(self.x_init)
        self.log_prop_values = self.log_prop_values.at[0:1, :].set(self.log_prop_fcn(self.x_init))
        uniform_rand = random.uniform(key=self.key, minval=0, maxval=1.0, shape=(self.iterations, self.n_chains))

        def main_algorithm(i: int = None) -> None:
            proposed = self.chains[:, :, i - 1] + rndw_samples[:, :, i]
            ln_prop = self.log_prop_fcn(proposed)
            hastings = jnp.minimum(jnp.exp(ln_prop - self.log_prop_values[i - 1, :]), 1)
            satis = (uniform_rand[i, :] < hastings)[0, :]
            non_satis = ~satis
            self.chains = self.chains.at[:, satis, i].set(proposed[:, satis])
            self.chains = self.chains.at[:, non_satis, i].set(self.chains[:, non_satis, i - 1])
            self.log_prop_values = self.log_prop_values.at[i, satis].set(ln_prop[0, satis])
            self.log_prop_values = self.log_prop_values.at[i, non_satis].set(self.log_prop_values[i - 1, non_satis])
            self.n_of_accept = self.n_of_accept.at[0, satis].set(self.n_of_accept[0, satis] + 1)
            self.accept_rate = self.accept_rate.at[i, :].set(self.n_of_accept[0, :] / i)
            return

        if not self.progress_bar:
            for i in tqdm(range(1, self.iterations), disable=self.progress_bar):
                main_algorithm(i)
        else:
            pass

        return self.chains[:, :, self.burnin:], self.accept_rate

    # def mhh_vectorized_sampling(self):
    #     """
    #     vectorized metropolis-hastings sampling algorithm used for sampling from the posteriori distribution
    #     :returns: chains: The chains of samples drawn from the posteriori distribution
    #               acceptance rate: The acceptance rate of the samples drawn form the posteriori distributions
    #     """
    #
    #     uniform_random_number = np.random.uniform(low=0.0, high=1.0, size=(self.Nchain, self.iterations))
    #
    #     for iteration in tqdm(range(1, self.iterations), disable=self.progress_bar):  # sampling from the distribution
    #         # generating the sample for each chain
    #         self.proposed = self.gaussian_proposed_distribution(self.chains[:, :, iteration - 1:iteration].copy(),
    #                                                             sigma=0.1)
    #         # calculating the log of the posteriori function
    #         Ln_prop = self.logprop_fcn(self.proposed, Covariance=1)
    #         # calculating the hasting ratio
    #         hastings = np.exp(Ln_prop - self.logprop[:, iteration - 1])
    #         criteria = uniform_random_number[ch, iteration] < hastings
    #         if criteria:
    #             self.chains[:, ch, iteration:iteration + 1] = self.proposed
    #             self.logprop[ch, iteration] = Ln_prop
    #             self.n_of_accept[ch, 0] += 1
    #             self.accept_rate[ch, iteration] = self.n_of_accept[ch, 0] / iteration
    #         else:
    #             self.chains[:, ch, iteration:iteration + 1] = self.chains[:, ch, iteration - 1: iteration]
    #             self.logprop[ch, iteration] = self.logprop[ch, iteration - 1]
    #             self.accept_rate[ch, iteration] = self.n_of_accept[ch, 0] / iteration
    #     return 1

# class MCMCHammer:
#     def __init__(self, logprop_fcn, iterations: int = None, rng: int = None, x0: jnp.ndarray = None,
#                  vectorized: bool = False,
#                  chains: int = 1, progress_bar: bool = True):
#
#         self.key = random.PRNGKey(rng)
#         # checking the correctness of log probability function
#         if hasattr(logprop_fcn, "__call__"):
#             self.logprop_fcn = logprop_fcn
#         else:
#             raise Exception('The log(probability) function is not defined properly!')
#
#         # checking the correctness of the iteration
#         if isinstance(iterations, int):
#             self.iterations = iterations
#         else:
#             self.iterations = 1000
#             print(
#                 f'--------------------------------------------------------------------------------------------------\n '
#                 f'The iteration is not an integer value.\n'
#                 f' The default value of {self.iterations} is selectd as the number of iterations\n'
#                 f'---------------------------------------------------------------------------------------------------')
#
#         # checking the correctness of the iteration
#         if isinstance(chains, int):
#             self.Nchain = chains
#         else:
#             self.Nchain = 1
#             print(
#                 f'---------------------------------------------------------------------------------------------------\n'
#                 f'The number of chains is not an integer value. '
#                 f'The default value of {self.Nchain} is selected as the number of chains\n'
#                 f'----------------------------------------------------------------------------------------------------')
#
#     def mcmc_hammer_non_vectorized_sampling(self):
#         random_uniform = random.uniform(key=self.key, minval=0, maxval=1.0, size=(self.n_chains, self.iterations))
#
#         def sample_proposal(a: float = None, chains: int = None, iterations: int = None):
#             random_uniform = random.uniform(key=self.key, minval=0, maxval=1.0)
#             return random_uniform * (jnp.sqrt(a) - jnp.sqrt(1 / a)) + jnp.sqrt(1 / a)
#
#         samples_of_gz = sample_proposal(a=2, chains=self.n_chains, iterations=self.iterations)
#
#         U_random = random.randint(self.C, size=(self.C, 1), dtype=int)


# logprop_fcn,
# logprop_fcn = Gaussian_liklihood,

#
# if __name__ == '__main__':
#     x0 = 15 * np.ones((1, 1))
#     x0 = np.tile(x0, (1, 5))
#     priori_distribution = dict()
# priori_distribution.update({'parameter1':})

# G = MetropolisHastings(logprop_fcn = gaussian_liklihood_single_variable, iterations=10000,
#                         x0 = x0, vectorized = False, chains=5, progress_bar=True)

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
# from jnx_Probablity_Distribution_Continuous import Uniform
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
#     # lg1 = theta1.log_pdf()
#     return 1
#
#
# nchains = 25
# theta_init = random.uniform(key=key, minval=0, maxval=1.0, shape=(len(theta), nchains))
#
# T = MetropolisHastings(log_prop_fcn=log_posteriori_function, model=model,
#                        iterations=150, chains=nchains, x_init=theta_init,
#                        progress_bar=True, burnin=30, activate_jit=False)
# T.sample()
