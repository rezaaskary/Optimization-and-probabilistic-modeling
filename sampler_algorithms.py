import numpy as np
import scipy as sc
from tqdm import tqdm
import matplotlib.pyplot as plt
from Probablity_distributions import *


class MetropolisHastings:
    def __init__(self, logprop_fcn, iterations: int = None, x_init: np.ndarray = None, vectorized: bool = False,
                 chains: int = 1, take_derivatives: bool = False, progress_bar: bool = True):

        # checking the correctness of log probability function
        if hasattr(logprop_fcn, "__call__"):
            self.logprop_fcn = logprop_fcn
        else:
            raise Exception('The log probability function is not defined properly!')

        # checking the correctness of the iteration
        if isinstance(iterations, int):
            self.iterations = iterations
        else:
            self.iterations = 1000
            print(f'-------------------------------------------------------------------------------------------------\n'
                  f'The iteration is not an integer value.\n'
                  f' The default value of {self.iterations} is selectd as the number of iterations\n'
                  f'--------------------------------------------------------------------------------------------------')

        # checking the correctness of the iteration
        if isinstance(chains, int):
            self.n_chains = chains
        else:
            self.n_chains = 1
            print(
                f'---------------------------------------------------------------------------------------------------\n'
                f'The number of chains is not an integer value.\n'
                f' The default value of {self.Nchain} is selectd as the number of chains\n'
                f'----------------------------------------------------------------------------------------------------')

        # checking the correctness of the vectorized simulation
        if isinstance(vectorized, bool):
            self.vectorized = vectorized
            if self.vectorized:
                self.run = self.mh_vectorized_sampling
            else:
                self.run = self.mh_non_vectorized_sampling
        else:
            self.vectorized = False
            self.run = self.mh_non_vectorized_sampling
            print(
                f'---------------------------------------------------------------------------------------------------\n'
                f'The default value of {self.vectorized} is selected for parallelized simulations\n'
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

        # checking the correctness of initial condition
        if isinstance(x_init, np.ndarray):
            dim1, dim2 = x_init.shape
            if dim2 != self.n_chains:
                raise Exception('The initial condition is not consistent with the number of chains!')
            else:
                self.ndim = dim1
                self.x_init = x_init
        else:
            raise Exception('The initial condition is not selected properly!')

        # initializing all values
        self.chains = np.zeros((self.ndim, self.n_chains, self.iterations))
        self.log_prop_values = np.zeros((self.n_chains, self.iterations))
        self.accept_rate = np.zeros((self.n_chains, self.iterations))
        self.log_prop_values[:, 0] = self.logprop_fcn(self.x0, Covariance=1)
        self.n_of_accept = np.zeros((self.n_chains, 1))

    def rw_parameter_proposal(self, x_old, sigma: float = 0.01):
        """
        :param x_old: the past values of adjustable parameters
        :param sigma: the standard deviation of the random walk model for proposing new set of values for parameters
        :return: new set of parameters (N)
        """
        x_old += np.random.randn(self.ndim, self.n_chains) * sigma
        return x_old

    def mh_non_vectorized_sampling(self):
        """
        non-vectorized metropolis-hastings sampling algorithm used for sampling from the posteriori distribution
        :returns: chains: The chains of samples drawn from the posteriori distribution
                  acceptance rate: The acceptance rate of the samples drawn form the posteriori distributions
        """
        # sampling from a uniform distribution
        uniform_random_number = np.random.uniform(low=0.0, high=1.0, size=(self.n_chains, self.iterations))

        for iter in tqdm(range(1, self.iterations), disable=self.progress_bar):       # sampling from the distribution
            for ch in (range(self.Nchain)):                                             # sampling from each cahin

                # generating the sample for each chain
                self.proposed = self.random_walk_parameter_proposal(self.chains[:, ch, iter-1:iter].copy(), sigma = 0.1)
                # calculating the log of the posteriori function
                Ln_prop = self.logprop_fcn(self.proposed, Covariance=1)
                # calculating the hasting ratio
                hastings = np.exp(Ln_prop - self.logprop[ch,iter-1])
                criteria = uniform_random_number[ch,iter]< hastings
                if criteria:
                    self.chains[:, ch, iter:iter+1] = self.proposed
                    self.logprop[ch, iter] = Ln_prop
                    self.n_of_accept[ch,0] += 1
                    self.accept_rate[ch,iter] = self.n_of_accept[ch,0] / iter
                else:
                    self.chains[:, ch, iter:iter+1] = self.chains[:, ch, iter - 1 : iter]
                    self.logprop[ch, iter] = self.logprop[ch, iter - 1]
                    self.accept_rate[ch, iter] = self.n_of_accept[ch,0] / iter

        # T1 = self.chains[0, 0, :]
        # plt.plot(T1)
        # plt.show()
        # plt.figure()
        # plt.plot(self.accept_rate.ravel())
        # plt.show()
        # T
        return self.chains, self.accept_rate

    def MH_vectorized_sampling(self):
        """
        vectorized metropolis-hastings sampling algorithm used for sampling from the posteriori distribution
        :returns: chains: The chains of samples drawn from the posteriori distribution
                  acceptance rate: The acceptance rate of the samples drawn form the posteriori distributions
        """

        # generating the uniform distribution to accept/or reject
        uniform_random_number = np.random.uniform(low=0.0, high=1.0, size=(self.Nchain, self.iterations))

        for iter in tqdm(range(1, self.iterations), disable=self.progress_bar):  # sampling from the distribution
            # generating the sample for each chain
            self.proposed = self.gaussian_proposed_distribution(self.chains[:, :, iter - 1:iter].copy(), sigma = 0.1)

            # calculating the log of the posteriori function
            Ln_prop = self.logprop_fcn(self.proposed, Covariance=1)
            # calculating the hasting ratio
            hastings = np.exp(Ln_prop - self.logprop[:, iter - 1])
            criteria = uniform_random_number[ch, iter] < hastings
            if criteria:
                self.chains[:, ch, iter:iter + 1] = self.proposed
                self.logprop[ch, iter] = Ln_prop
                self.n_of_accept[ch, 0] += 1
                self.accept_rate[ch, iter] = self.n_of_accept[ch, 0] / iter
            else:
                self.chains[:, ch, iter:iter + 1] = self.chains[:, ch, iter - 1: iter]
                self.logprop[ch, iter] = self.logprop[ch, iter - 1]
                self.accept_rate[ch, iter] = self.n_of_accept[ch, 0] / iter
        return 1


class MCMCHammer:
    def __init__(self, logprop_fcn, iterations: int = None, x0: np.ndarray = None, vectorized:bool = False,
                 chains: int = 1, progress_bar: bool = True):

        # checking the correctness of log probability function
        if hasattr(logprop_fcn, "__call__"):
            self.logprop_fcn = logprop_fcn
        else:
            raise Exception('The log(probability) function is not defined properly!')

        # checking the correctness of the iteration
        if isinstance(iterations, int):
            self.iterations = iterations
        else:
            self.iterations = 1000
            print(
                f'--------------------------------------------------------------------------------------------------\n '
                f'The iteration is not an integer value.\n'
                f' The default value of {self.iterations} is selectd as the number of iterations\n'
                f'---------------------------------------------------------------------------------------------------')

        # checking the correctness of the iteration
        if isinstance(chains, int):
            self.Nchain = chains
        else:
            self.Nchain = 1
            print(
                f'------------------------------------------------------------------------------------------------------------------\n '
                f'The number of chains is not an integer value. The default value of {self.Nchain} is selectd as the number of chains\n'
                f'---------------------------------------------------------------------------------------------------------------------')


# logprop_fcn,
# logprop_fcn = Gaussian_liklihood,


if __name__=='__main__':
    x0 = 15 * np.ones((1, 1))
    x0 = np.tile(x0,(1,5))
    priori_distribution = dict()
    # priori_distribution.update({'parameter1':})

    G = Metropolis_Hastings(logprop_fcn = gaussian_liklihood_single_variable, iterations=10000,
                            x0 = x0, vectorized = False, chains=5, progress_bar=True)
    G.run()