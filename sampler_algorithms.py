import numpy as np
import scipy as sc
from tqdm import tqdm
import matplotlib.pyplot as plt
from Probablity_distributions import *
# import emcee
# from emcee import EnsembleSampler


def gaussian_liklihood_single_variable(measured:np.ndarray, estimated: np.ndarray, N:int, sigma: np.ndarray) -> np.ndarray:
    """
    The single variable Gausian liklihood function
    :param measured: The measured variable
    :param estimated: The estimated variable(or calculated from a model)
    :param N: The number of measured samples
    :param sigma: The standard deviation of the error estimation
    :return: the log_liklihood function
    """
    log_gauss = -N*np.log(sigma * np.sqrt(2 * np.pi)) - (((measured - estimated) ** 2) / (2 * sigma ** 2)).sum()
    return log_gauss
    #====================================================================================

def gaussian_liklihood_single_variable_vectorized(measured: np.ndarray, estimated: np.ndarray, N: int,C: int, sigma: np.ndarray) -> np.ndarray:
    """
    The single variable Gausian liklihood function
    :param measured: The measured variable (NxC)
    :param estimated: The estimated variable or calculated from a model (NxC)
    :param N: An integer indicating the number of measured samples
    :param C: An integer indicating the number of Chains
    :param sigma: The standard deviation of the error estimation (1xC)
    :return: A numpy array indicating the log_liklihood function (1xC)
    """
    vectorized_error = ((measured - estimated)**2).sum(axis=0)
    log_gauss = -N*np.log(sigma * np.sqrt(2 * np.pi)) - (vectorized_error / (2 * sigma ** 2))
    return log_gauss
    #====================================================================================






def gaussian_liklihood_multivariable(measured:np.ndarray, estimated:np.ndarray, N:int, Covariance: np.ndarray, K:int) -> np.ndarray:
    """
    The log liklihood of the Multivariable gaussian distribution used for multivariable fitting (multivariables objective function)
    :param measured: KxN measured parameters (K dimentional parameters and N sampling points)
    :param estimated:KxN estimated parameters (K dimentional parameters and N sampling points)
    :param N: An integer indicating the number of measurements
    :param Covariance: A positive definite square matrix indicating the covariance matrix of the multivariable Normal distribution (KxK)
    :param K: The dimention of the multivariable gaussian distribution
    :return: The log liklihood of the
    """

    inv_cov = np.linalg.inv(Covariance)
    det_cov = np.linalg.det(Covariance)
    Error = measured - estimated        # KxN error matrix

    log_gauss = -N * np.log(np.sqrt(((2 * np.pi)**K) * det_cov)) - (0.5 * (np.diag(Error.T @ inv_cov @ Error)) ).sum()
    return log_gauss


def gaussian_liklihood_vectorized(parameter: np.ndarray, Covariance: np.ndarray = np.eye(1))->np.ndarray:
    return


def gaussian_lihlihood_vectorized(parameter, n:int = 1) -> np.ndarray:

    return 1



class Metropolis_Hastings:
    def __init__(self,logprop_fcn, iterations:int = None, x0:np.ndarray = None, vectorized:bool = False, chains:int = 1, progress_bar:bool = True):

        # checking the correctness of log probablity function
        if hasattr(logprop_fcn,"__call__"):
            self.logprop_fcn = logprop_fcn
        else:
            raise  Exception('The log(probablity) function is not defined probperly!')

        # checking the correctness of the iteration
        if isinstance(iterations, int):
            self.iterations = iterations
        else:
            self.iterations = 1000
            print(f'------------------------------------------------------------------------------------------------------------------\n '
                  f'The iteration is not an integer value. The default value of {self.iterations} is selectd as the number of iterations\n'
                  f'---------------------------------------------------------------------------------------------------------------------')

        # checking the correctness of the iteration
        if isinstance(chains, int):
            self.Nchain = chains
        else:
            self.Nchain = 1
            print(
                f'------------------------------------------------------------------------------------------------------------------\n '
                f'The number of chains is not an integer value. The default value of {self.Nchain} is selectd as the number of chains\n'
                f'---------------------------------------------------------------------------------------------------------------------')

        # checking the correctness of the vectorized simulation
        if isinstance(vectorized, bool):
            self.vectorized = vectorized
            if self.vectorized:
                self.run = self.MH_vectorized_sampling
            else:
                self.run = self.MH_non_vectorized_sampling
        else:
            self.vectorized = False
            self.run = self.MH_non_vectorized_sampling
            print(
                f'------------------------------------------------------------------------------------------------------------------\n '
                f'The default value of {self.vectorized} is selectd for vectorizing simulations\n'
                f'---------------------------------------------------------------------------------------------------------------------')

        # checking the corectness of the progressbar
        if isinstance(progress_bar, bool):
            self.progress_bar = not progress_bar
        else:
            self.progress_bar = False
            print(
                f'------------------------------------------------------------------------------------------------------------------\n '
                f'The progress bar is activated by default since the it is not entered by the user\n'
                f'---------------------------------------------------------------------------------------------------------------------')

        # checking the correctness of initial condition
        if isinstance(x0, np.ndarray):
            dim1, dim2 = x0.shape
            if dim2 != self.Nchain:
                raise Exception('The initial condition is not consistent with the number of chains!')
            else:
                self.Ndim = dim1
                self.x0 = x0
        else:
            raise Exception('The initial condition is not selected properly!')

        # initializing all values
        self.chains = np.zeros((self.Ndim, self.Nchain, self.iterations))
        self.logprop = np.zeros((self.Nchain, self.iterations))
        self.accept_rate = np.zeros((self.Nchain, self.iterations))
        self.logprop[:, 0] = self.logprop_fcn(self.x0, Covariance=1)
        self.n_of_accept = np.zeros((self.Nchain, 1))


    def MH_non_vectorized_sampling(self):
        """
        non-vectorized metropolis-hastings sampling algorithm used for sampling from the posteriori distribution
        :returns: chains: The chains of samples drawn from the posteriori distribution
                  acceptance rate: The acceptance rate of the samples drawn form the posteriori distributions
        """
        # sampling from a uniform distribution
        uniform_random_number = np.random.uniform(low = 0.0, high = 1.0, size=(self.Nchain, self.iterations))

        for iter in tqdm(range(1, self.iterations), disable = self.progress_bar):       # sampling from the distribution
            for ch in (range(self.Nchain)):                                             # sampling from each cahin

                # generating the sample for each chain
                self.proposed = self.gaussian_proposed_distribution(self.chains[:, ch, iter-1:iter].copy(), sigma = 0.1)
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
            self.proposed = self.gaussian_proposed_distribution(self.chains[:, ch, iter - 1:iter].copy(), sigma=0.1)
            # calculating the log of the posteriori function
            Ln_prop = self.logprop_fcn(self.proposed, Covariance=1)
            # calculating the hasting ratio
            hastings = np.exp(Ln_prop - self.logprop[ch, iter - 1])
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



    def gaussian_proposed_distribution(self, x_old, sigma:float = 0.01):
        x_old += np.random.randn(self.Ndim, self.Nchain) * sigma
        return x_old


def Gaussian_liklihood(parameter):
    x = parameter[0:1,0]
    # x = parameter[1:,0]
    mean = 0
    sigma = 2
    log_gauss = -np.log(sigma * np.sqrt(2 * np.pi)) - ((x - mean) ** 2) / (2 * sigma ** 2)
    return log_gauss


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