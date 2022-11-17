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


class ParameterProposalInitialization:
    def __init__(self, log_prop_fcn: callable = None,
                 iterations: int = None,
                 burnin: int = None,
                 x_init: jnp.ndarray = None,
                 activate_jit: bool = False,
                 chains: int = 1,
                 progress_bar: bool = True,
                 random_seed: int = 1,
                 move: str = 'single_stretch',
                 cov: jnp.ndarray = None,
                 n_split: int = 2,
                 a: float = None):

        if isinstance(move, str):  # checking the correctness of parameter proposal algorithm
            if move in ['single_stretch', 'random_walk', 'parallel_stretch']:
                self.move = move
        elif not move:
            self.move = 'random_walk'
        else:
            raise Exception('The algorithm of updating proposal parameters is not specified correctly')

        if hasattr(log_prop_fcn, "__call__"):  # checking the correctness of log probability function
            self.log_prop_fcn = log_prop_fcn
        else:
            raise Exception('The log probability function is not defined properly!')

        self.key = random.PRNGKey(random_seed)

        if isinstance(iterations, int):  # checking the correctness of the iteration
            self.iterations = iterations
        else:
            self.iterations = 1000
            print(f'-------------------------------------------------------------------------------------------------\n'
                  f'The iteration is not an integer value.\n'
                  f' The default value of {self.iterations} is selected as the number of iterations\n'
                  f'--------------------------------------------------------------------------------------------------')

        if isinstance(burnin, int):  # checking the correctness of the burnin period
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

        if self.burnin >= self.iterations:  # checking the correctness of iteration and burnin period
            raise Exception('The number of samples selected for burnin cannot be greater than the simulation samples!')

        if isinstance(chains, int):  # checking the correctness of the number of chains
            self.n_chains = chains
        else:
            self.n_chains = 1
            print(
                f'---------------------------------------------------------------------------------------------------\n'
                f'The number of chains is not an integer value.\n'
                f' The default value of {self.n_chains} is selected as the number of chains\n'
                f'----------------------------------------------------------------------------------------------------')

        if isinstance(n_split, int):  # checking the correctness of the number of split
            if self.move == 'parallel_stretch' and self.n_chains // n_split:
                raise Exception(f'The number of chains should be a multiplication of the number of splits.\n'
                                f'As a suggestion, you may use {((self.n_chains // n_split) + 1) * n_split} as the number\n'
                                f'of chains.')
            if self.move == 'parallel_stretch' and not self.n_chains // n_split:
                self.n_split = n_split
                self.split_len = self.n_chains // self.n_split
        elif not n_split:
            self.n_split = 2
            print(
                f'---------------------------------------------------------------------------------------------------\n'
                f'The number of solit is not specified.\n'
                f' The default value of {self.n_split} is selected as the number of splits\n'
                f'----------------------------------------------------------------------------------------------------')
        else:
            raise Exception('The number of splits for ensemble sampling is not specified correctly')

        if isinstance(x_init, jnp.ndarray):  # checking the correctness of initial condition
            dim1, dim2 = x_init.shape
            if dim2 != self.n_chains:
                raise Exception('The initial condition is not consistent with the number of chains!')
            elif dim1 * 2 > self.n_chains:
                raise Exception('The number of chains should be least two times of the dimension of the parameters')
            else:
                self.ndim = dim1
                self.x_init = x_init
        else:
            raise Exception('The initial condition is not selected properly!')

        if isinstance(activate_jit, bool):  # checking the correctness of the just-in-time simulation
            self.activate_jit = activate_jit
        else:
            self.activate_jit = False
            print(
                f'---------------------------------------------------------------------------------------------------\n'
                f'The default value of {self.activate_jit} is selected for parallelized simulations\n'
                f'----------------------------------------------------------------------------------------------------')

        if isinstance(progress_bar, bool):  # checking the correctness of the progressbar
            self.progress_bar = not progress_bar
        else:
            self.progress_bar = False
            print(
                f'---------------------------------------------------------------------------------------------------\n'
                f'The progress bar is activated by default since the it is not entered by the user\n'
                f'----------------------------------------------------------------------------------------------------')

        if isinstance(cov, jnp.ndarray):  # checking the correctness of the covariance
            if (cov.shape[0] != cov.shape[1]) or (cov.shape[0] != self.ndim):
                raise Exception('The size of the covariance matrix is either incorrect or inconsistent with the'
                                ' dimension of the parameters')
            else:
                if jnp.all(jnp.linalg.eigvals(cov) > 0):
                    self.cov_proposal = cov
                else:
                    raise Exception('The covariance matrix of updating parameters should be positive definite')
        elif not cov:
            self.cov_proposal = None
        else:
            raise Exception('The covariance matrix for calculating proposal parameters are not entered correctly')

        if isinstance(a,
                      (float, int)):  # checking the correctness of the scaling factor a (for single/parallel stretch)
            if a > 1:
                self.a_proposal = a
            else:
                raise Exception('The value of a should be greater than 1')
        elif not a:
            self.a_proposal = None
        else:
            raise Exception('The value of a is not specified correctly')

        if self.move == 'random_walk':  # using random walk proposal algorithm
            self.rndw_samples = jnp.transpose(random.multivariate_normal(key=self.key, mean=jnp.zeros((1, self.ndim)),
                                                                         cov=self.cov_proposal[jnp.newaxis, :, :],
                                                                         shape=(self.iterations, self.n_chains)),
                                              axes=(2, 1, 0))
            self.proposal_alg = self.random_walk_proposal

        elif self.move == 'single_stretch':  # using single stretch proposal algorithm
            self.z = jnp.power(
                (random.uniform(key=self.key, minval=0, maxval=1.0, shape=(self.iterations, self.n_chains)) *
                 (jnp.sqrt(self.a_proposal) - jnp.sqrt(1 / self.a_proposal)) + jnp.sqrt(1 / self.a_proposal)),
                2)
            self.index = jnp.zeros((self.iterations, self.n_chains), dtype=int)
            ordered_index = jnp.arange(self.n_chains, dtype=int)
            for i in range(self.n_chains):
                self.index = self.index.at[:, i].set(random.choice(key=self.key, a=jnp.delete(arr=ordered_index, obj=i),
                                                                   replace=True, shape=(self.iterations,)))
                self.key += 1
            self.proposal_alg = self.single_strech_move

        elif self.move == 'parallel_stretch':  # using parallel stretch move parameter proposal by dividing chains into
            # n_split sub walkers
            self.iterations *= 2
            self.z = jnp.power(
                (random.uniform(key=self.key, minval=0, maxval=1.0, shape=(self.iterations, self.n_chains)) *
                 (jnp.sqrt(self.a_proposal) - jnp.sqrt(1 / self.a_proposal)) + jnp.sqrt(1 / self.a_proposal)),
                2)

            ordered_index = jnp.arange(self.n_split).astype(int)
            single_split = jnp.arange(start=0, step=1, stop=self.split_len)
            for i in range(self.n_split):
                selected_split = random.choice(key=self.key, a=jnp.delete(arr=ordered_index, obj=i), replace=True,
                                               shape=(self.iterations, 1))
                self.index = self.index.at[:, i * self.split_len:(i + 1) * self.split_len].set(random.permutation(
                    key=self.key,
                    x=selected_split * self.split_len + single_split,
                    axis=1,
                    independent=True))
                self.key += 1
            self.proposal_alg = self.single_strech_move
        else:
            raise Exception('A method for updating parameters should be entered')

    def random_walk_proposal(self, whole_chains: jnp.ndarray = None, itr: int = None):
        return whole_chains[:, :, itr - 1] + self.rndw_samples[:, :, itr - 1]

    def single_strech_move(self, whole_chains: jnp.ndarray = None, itr: int = None):
        return whole_chains[:, self.index[itr - 1, :], itr - 1] + self.z[itr, :] * (
                whole_chains[:, :, itr - 1] - whole_chains[:, self.index[itr - 1, :], itr - 1])


class MetropolisHastings(ParameterProposalInitialization):
    def __init__(self, log_prop_fcn: callable = None, iterations: int = None, burnin: int = None,
                 x_init: jnp.ndarray = None, activate_jit: bool = False, chains: int = 1, progress_bar: bool = True,
                 random_seed: int = 1, cov: jnp.ndarray = None):
        """
        Metropolis Hastings sampling algorithm
        :param log_prop_fcn: Takes the log posteriori function
        :param iterations: The number of iteration
        :param burnin: The number of initial samples to be droped sowing to non-stationary behaviour
        :param x_init: The initialized value of parameters
        :param parallelized: A boolean variable used to activate or deactivate the parallelized calculation
        :param chains: the number of chains used for simulation
        :param progress_bar: A boolean variable used to activate or deactivate the progress bar. Deactivation of the
         progress bar results in activating XLA -accelerated iteration for the fast  evaluation  of the
          chains(recommended!)
        :param model: The model function (a function that input parameters and returns estimations)
        """
        super(MetropolisHastings, self).__init__(log_prop_fcn=log_prop_fcn, iterations=iterations, burnin=burnin,
                                                 x_init=x_init, activate_jit=activate_jit, chains=chains, cov=cov,
                                                 progress_bar=progress_bar, random_seed=random_seed, move='random_walk')

        # initializing chain values
        self.chains = jnp.zeros((self.ndim, self.n_chains, self.iterations))
        # initializing the log of the posteriori values
        self.log_prop_values = jnp.zeros((self.iterations, self.n_chains))
        # initializing the track of hasting ratio values
        self.accept_rate = jnp.zeros((self.iterations, self.n_chains))
        # in order to calculate the acceptance ration of all chains
        self.n_of_accept = jnp.zeros((1, self.n_chains))

    def sample(self):
        """
        vectorized metropolis-hastings sampling algorithm used for sampling from the posteriori distribution
        :returns: chains: The chains of samples drawn from the posteriori distribution
                  acceptance rate: The acceptance rate of the samples drawn form the posteriori distributions
        """
        self.chains = self.chains.at[:, :, 0].set(self.x_init)
        self.log_prop_values = self.log_prop_values.at[0:1, :].set(self.log_prop_fcn(self.x_init))
        self.uniform_rand = random.uniform(key=self.key, minval=0, maxval=1.0, shape=(self.iterations, self.n_chains))

        def alg_with_progress_bar(itr: int = None) -> None:
            # The function suited for using progress bar
            proposed = self.proposal_alg(whole_chains=self.chains, itr=itr)
            ln_prop = self.log_prop_fcn(proposed)
            hastings = jnp.minimum(jnp.exp(ln_prop - self.log_prop_values[itr - 1, :]), 1)
            satis = (self.uniform_rand[itr, :] < hastings)[0, :]
            non_satis = ~satis
            self.chains = self.chains.at[:, satis, itr].set(proposed[:, satis])
            self.chains = self.chains.at[:, non_satis, itr].set(self.chains[:, non_satis, itr - 1])
            self.log_prop_values = self.log_prop_values.at[itr, satis].set(ln_prop[0, satis])
            self.log_prop_values = self.log_prop_values.at[itr, non_satis].set(self.log_prop_values[i - 1, non_satis])
            self.n_of_accept = self.n_of_accept.at[0, satis].set(self.n_of_accept[0, satis] + 1)
            self.accept_rate = self.accept_rate.at[itr, :].set(self.n_of_accept[0, :] / itr)
            return

        def alg_with_lax_acclelrated(itr: int, recursive_variables: tuple) -> tuple:
            # The function suited for fast and efficient evaluation of chains
            lax_chains, lax_log_prop_values, lax_n_of_accept, lax_accept_rate = recursive_variables
            proposed = self.proposal_alg(whole_chains=lax_chains, itr=itr)
            ln_prop = self.log_prop_fcn(proposed)
            hastings = jnp.minimum(jnp.exp(ln_prop - lax_log_prop_values[itr - 1, :]), 1)
            lax_log_prop_values = lax_log_prop_values.at[itr, :].set(jnp.where(self.uniform_rand[itr, :] < hastings,
                                                                               ln_prop,
                                                                               lax_log_prop_values[itr - 1, :])[0, :])
            lax_chains = lax_chains.at[:, :, itr].set(jnp.where(self.uniform_rand[itr, :] < hastings,
                                                                proposed,
                                                                lax_chains[:, :, itr - 1]))
            lax_n_of_accept = lax_n_of_accept.at[0, :].set(jnp.where(self.uniform_rand[itr, :] < hastings,
                                                                     lax_n_of_accept[0, :] + 1,
                                                                     lax_n_of_accept[0, :])[0, :])
            lax_accept_rate = lax_accept_rate.at[itr, :].set(lax_n_of_accept[0, :] / itr)
            return lax_chains, lax_log_prop_values, lax_n_of_accept, lax_accept_rate

        if not self.progress_bar:
            for i in tqdm(range(1, self.iterations), disable=self.progress_bar):
                alg_with_progress_bar(i)
        else:
            print('Simulating...')
            self.chains, \
            self.log_prop_values, \
            self.n_of_accept, \
            self.accept_rate = lax.fori_loop(lower=1,
                                             upper=self.iterations,
                                             body_fun=alg_with_lax_acclelrated,
                                             init_val=(
                                                 self.chains.copy(),
                                                 self.log_prop_values.copy(),
                                                 self.n_of_accept.copy(),
                                                 self.accept_rate.copy()
                                             ))
        return self.chains[:, :, self.burnin:], self.accept_rate


class MCMCHammer(ParameterProposalInitialization):
    def __init__(self, log_prop_fcn: callable = None, iterations: int = None, burnin: int = None,
                 x_init: jnp.ndarray = None, activate_jit: bool = False, chains: int = 1, progress_bar: bool = True,
                 random_seed: int = 1, move: str = 'single_stretch', a: float = 2):
        """
        MCMC Hammer empowered with jax to large scale simulation
        :param log_prop_fcn: A callable function returning the log-likelihood (or posteriori) of the distribution
        :param iterations: An integer indicating the number of steps(or samples)
        :param burnin: An integer used for truncating chains of samples to remove the transient variation of chains
        :param x_init: An matrix (NxC) encompassing the initial condition for each chain
        :param activate_jit: A boolean variable for activating/deactivating just-in-time evaluation of functions
        :param chains: An integer determining the number of chains
        :param progress_bar: A boolean variable used for activating or deactivating the progress bar
        :param random_seed: An integer for fixing rng
        :param move: A string variable used to determine the algorithm for calculating the proposal parameters. Options
        are "single_stretch", "parallel_stretch"
        :param a: An adjustable scale parameter (1<a) used for calculating the proposal parameters
        """
        super(MCMCHammer, self).__init__(log_prop_fcn=log_prop_fcn, iterations=iterations, burnin=burnin,
                                         x_init=x_init, activate_jit=activate_jit, chains=chains,
                                         progress_bar=progress_bar, random_seed=random_seed, move=move, a=a)

        # initializing chain values
        self.chains = jnp.zeros((self.ndim, self.n_chains, self.iterations))
        # initializing the log of the posteriori values
        self.log_prop_values = jnp.zeros((self.iterations, self.n_chains))
        # initializing the track of hasting ratio values
        self.accept_rate = jnp.zeros((self.iterations, self.n_chains))
        # in order to calculate the acceptance ration of all chains
        self.n_of_accept = jnp.zeros((1, self.n_chains))

    def sample(self):
        """
        vectorized MCMC Hammer sampling algorithm used for sampling from the posteriori distribution. Developed based on
         the paper published in 2013:
         <<Foreman-Mackey, Daniel, et al. "emcee: the MCMC hammer." Publications of the Astronomical
          Society of the Pacific 125.925 (2013): 306.>>
        :returns: chains: The chains of samples drawn from the posteriori distribution
                  acceptance rate: The acceptance rate of the samples drawn form the posteriori distributions
        """
        self.chains = self.chains.at[:, :, 0].set(self.x_init)
        self.log_prop_values = self.log_prop_values.at[0:1, :].set(self.log_prop_fcn(self.x_init))
        self.uniform_rand = random.uniform(key=self.key, minval=0, maxval=1.0, shape=(self.iterations, self.n_chains))

        # # for single streatch
        # self.index = jnp.zeros((self.iterations, self.n_chains))
        # ordered_index = jnp.arange(self.n_chains).astype(int)
        # for i in range(self.n_chains):
        #     self.index = self.index.at[:, i].set(random.choice(key=self.key, a=jnp.delete(arr=ordered_index, obj=i),
        #                                                        replace=True, shape=(self.iterations,)))
        #
        # n_split = 4
        # self.n_split = n_split
        # self.split_len = self.n_chains // self.n_split
        # ordered_index = jnp.arange(self.n_split).astype(int)
        # single_split = jnp.arange(start=0, step=1, stop=self.split_len)
        # for i in range(self.n_split):
        #     selected_split = random.choice(key=self.key, a=jnp.delete(arr=ordered_index, obj=i), replace=True,
        #                                    shape=(self.iterations, 1))
        #     # XX = random.permutation(key=self.key, x=selected_split * self.split_len + single_split, axis=1, independent=True)
        #     self.index = self.index.at[:, i * self.split_len:(i + 1) * \
        #                                                      self.split_len].set(random.permutation(key=self.key,
        #                                                                                             x=selected_split * self.split_len + single_split,
        #                                                                                             axis=1,
        #                                                                                             independent=True))
        #     self.key += 1

        def alg_with_progress_bar(itr: int = None) -> None:
            # The function suited for using progress bar
            proposed = self.proposal_alg(whole_chains=self.chains, itr=itr)
            ln_prop = self.log_prop_fcn(proposed)
            hastings = jnp.minimum(jnp.power(self.z[itr - 1, :], self.ndim - 1) *
                                   jnp.exp(ln_prop - self.log_prop_values[itr - 1, :]), 1)
            satis = (self.uniform_rand[itr, :] < hastings)[0, :]
            non_satis = ~satis
            self.chains = self.chains.at[:, satis, itr].set(proposed[:, satis])
            self.chains = self.chains.at[:, non_satis, itr].set(self.chains[:, non_satis, itr - 1])
            self.log_prop_values = self.log_prop_values.at[itr, satis].set(ln_prop[0, satis])
            self.log_prop_values = self.log_prop_values.at[itr, non_satis].set(self.log_prop_values[i - 1, non_satis])
            self.n_of_accept = self.n_of_accept.at[0, satis].set(self.n_of_accept[0, satis] + 1)
            self.accept_rate = self.accept_rate.at[itr, :].set(self.n_of_accept[0, :] / itr)
            return

        def alg_with_lax_acclelrated(itr: int, recursive_variables: tuple) -> tuple:
            # The function suited for fast and efficient evaluation of chains
            lax_chains, lax_log_prop_values, lax_n_of_accept, lax_accept_rate = recursive_variables
            proposed = self.proposal_alg(whole_chains=lax_chains, itr=itr)
            ln_prop = self.log_prop_fcn(proposed)
            hastings = jnp.minimum(jnp.power(self.z[itr - 1, :], self.ndim - 1) *
                                   jnp.exp(ln_prop - lax_log_prop_values[itr - 1, :]), 1)
            lax_log_prop_values = lax_log_prop_values.at[itr, :].set(jnp.where(self.uniform_rand[itr, :] < hastings,
                                                                               ln_prop,
                                                                               lax_log_prop_values[itr - 1, :])[0, :])
            lax_chains = lax_chains.at[:, :, itr].set(jnp.where(self.uniform_rand[itr, :] < hastings,
                                                                proposed,
                                                                lax_chains[:, :, itr - 1]))
            lax_n_of_accept = lax_n_of_accept.at[0, :].set(jnp.where(self.uniform_rand[itr, :] < hastings,
                                                                     lax_n_of_accept[0, :] + 1,
                                                                     lax_n_of_accept[0, :])[0, :])
            lax_accept_rate = lax_accept_rate.at[itr, :].set(lax_n_of_accept[0, :] / itr)
            return lax_chains, lax_log_prop_values, lax_n_of_accept, lax_accept_rate

        # def parallel_streatch_indexing(itr, inputs) -> tuple:
        #     lax_index, lax_single_split, key = inputs
        #     ordered_index = jnp.arange(self.n_split).astype(int)
        #     vd = jnp.setdiff1d(ordered_index, itr)
        #     selected_split = random.choice(key=key, a=vd, replace=True,
        #                                    shape=(self.iterations, 1))
        #     premuted_split = random.permutation(key=key, x=selected_split * self.split_len + lax_single_split,
        #                                         axis=1,
        #                                         independent=True)
        #     lax_index = lax_index.at[:, i * self.split_len:(i + 1) * self.split_len].set(premuted_split)
        #     key += 1
        #     return lax_index, key
        #
        # VV = lax.fori_loop(lower=0,
        #                    upper=self.n_split,
        #                    body_fun=parallel_streatch_indexing,
        #                    init_val=(self.index,
        #                              single_split,
        #                              self.key))
        #
        # self.chains = self.chains.at[:, :, 0].set(self.x_init)
        # self.log_prop_values = self.log_prop_values.at[0:1, :].set(self.log_prop_fcn(self.x_init))
        # uniform_rand = random.uniform(key=self.key, minval=0, maxval=1.0, shape=(self.iterations, self.n_chains))
        # a = 2
        # z = jnp.power((random.uniform(key=self.key, minval=0, maxval=1.0, shape=(self.iterations, self.n_chains)) *
        #                (jnp.sqrt(a) - jnp.sqrt(1 / a)) + jnp.sqrt(1 / a)), 2)

        # def alg_with_progress_bar(i: int = None) -> None:
        #     """
        #     main algorithm of Ensemble MCMC Hammer (Algorithm 2: A single stretch move update step)
        #     :param i: The iterator
        #     :return:  None
        #     """
        #     proposed = self.chains[:, self.index[i - 1, :], i - 1] + z[i - 1, :] * (
        #             self.chains[:, :, i - 1] - self.chains[:, self.index[i - 1, :], i - 1])
        #     ln_prop = self.log_prop_fcn(proposed)
        #     hastings = jnp.minimum(jnp.power(z[i - 1, :], self.ndim - 1) *
        #                            jnp.exp(ln_prop - self.log_prop_values[i - 1, :]), 1)
        #     satis = (uniform_rand[i, :] < hastings)[0, :]
        #     non_satis = ~satis
        #     self.chains = self.chains.at[:, satis, i].set(proposed[:, satis])
        #     self.chains = self.chains.at[:, non_satis, i].set(self.chains[:, non_satis, i - 1])
        #     self.log_prop_values = self.log_prop_values.at[i, satis].set(ln_prop[0, satis])
        #     self.log_prop_values = self.log_prop_values.at[i, non_satis].set(self.log_prop_values[i - 1, non_satis])
        #     self.n_of_accept = self.n_of_accept.at[0, satis].set(self.n_of_accept[0, satis] + 1)
        #     self.accept_rate = self.accept_rate.at[i, :].set(self.n_of_accept[0, :] / i)
        #     return

        # def alg_with_lax_acclelrated(i: int, recursive_variables: tuple) -> tuple:
        #     """
        #     lax-accelerated main algorithm of Ensemble MCMC Hammer (Algorithm 2: A single stretch move update step)
        #     :param i: The iterator
        #     :return:  None
        #     """
        #     lax_chains, lax_log_prop_values, lax_n_of_accept, lax_accept_rate, lax_indexed = recursive_variables
        #     proposed = lax_chains[:, lax_indexed[i - 1, :], i - 1] + z[i - 1, :] * (
        #             lax_chains[:, :, i - 1] - lax_chains[:, lax_indexed[i - 1, :], i - 1])
        #     ln_prop = self.log_prop_fcn(proposed)
        #
        #     hastings = jnp.minimum(jnp.power(z[i - 1, :], self.ndim - 1) *
        #                            jnp.exp(ln_prop - lax_log_prop_values[i - 1, :]), 1)
        #
        #     lax_log_prop_values = lax_log_prop_values.at[i, :].set(jnp.where(uniform_rand[i, :] < hastings,
        #                                                                      ln_prop,
        #                                                                      lax_log_prop_values[i - 1, :])[0, :])
        #     lax_chains = lax_chains.at[:, :, i].set(jnp.where(uniform_rand[i, :] < hastings,
        #                                                       proposed,
        #                                                       lax_chains[:, :, i - 1]))
        #     lax_n_of_accept = lax_n_of_accept.at[0, :].set(jnp.where(uniform_rand[i, :] < hastings,
        #                                                              lax_n_of_accept[0, :] + 1,
        #                                                              lax_n_of_accept[0, :])[0, :])
        #     lax_accept_rate = lax_accept_rate.at[i, :].set(lax_n_of_accept[0, :] / i)
        #     return (lax_chains, lax_log_prop_values, lax_n_of_accept, lax_accept_rate, lax_indexed)

        # Calculating proposal samples from the main group of the  samples

        if not self.progress_bar:
            for i in tqdm(range(1, self.iterations), disable=self.progress_bar):
                alg_with_progress_bar(itr=i)
        else:
            print('Simulating...')
            self.chains, \
            self.log_prop_values, \
            self.n_of_accept, \
            self.accept_rate = lax.fori_loop(lower=1,
                                             upper=self.iterations,
                                             body_fun=alg_with_lax_acclelrated,
                                             init_val=(
                                                 self.chains.copy(),
                                                 self.log_prop_values.copy(),
                                                 self.n_of_accept.copy(),
                                                 self.accept_rate.copy()
                                             ))

        return self.chains[:, :, self.burnin:], self.accept_rate


class HMC:
    def __init__(self, log_prop_fcn: callable = None, iterations: int = None, burnin: int = None,
                 x_init: jnp.ndarray = None, activate_jit: bool = False, chains: int = 1, progress_bar: bool = True,
                 random_seed: int = 1, move: str = 'single_stretch'):
        """

        :param log_prop_fcn:
        :param iterations:
        :param burnin:
        :param x_init:
        :param activate_jit:
        :param chains:
        :param progress_bar:
        :param random_seed:
        :param move:
        """
        super(HMC, self).__init__(log_prop_fcn=log_prop_fcn, iterations=iterations, burnin=burnin,
                                  x_init=x_init, activate_jit=activate_jit, chains=chains,
                                  progress_bar=progress_bar, random_seed=random_seed, move=move)
