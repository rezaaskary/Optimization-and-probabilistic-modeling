import numpy as np
import scipy as sc
from tqdm import tqdm
import emcee
from emcee import EnsembleSampler



class Metropolis_Hastings:
    def __int__(self, logprop_fcn, iterations:int = 1000, x0:np.ndarray = np.ones((1,1)), vectorized:bool = False, chains:int = 1):
        self.iterations = iterations
        self.x0 = x0
        self.Ndim = len(self.x0)
        self.chains = chains
        self.chains = np.zeros((self.Ndim, self.chains, self.iterations))
        self.logprop_fcn = logprop_fcn
        self.logprop = np.zeros((self.chains, self.iterations))
        self.accept_rate = np.zeros((self.chains, self.iterations))
        self.chains[:, :, 0] = self.x0
        self.logprop[:,0] = self.logprop_fcn(self.x0)

    def MH_non_vectorized_sampling(self):
        uniform_random_number = random.uniform(low = 0.0, high = 1.0, size = 1)
        for ch in tqdm(range(self.chains)):
            self.n_of_accept = 0
            for iter in range(1,self.iterations):
                self.proposed = self.gaussian_proposal(self.chains[:, :, iter-1],sigma = 0.1)

                hastings = np.exp(self.logprop_fcn(self.proposed))/np.exp(self.logprop[:,iter-1])
                criteria = uniform_random_number < hastings
                if criteria:
                    self.chains[:, ch, iter] = self.proposed
                    self.n_of_accept += 1
                    self.accept_rate[ch,iter] = self.n_of_accept / iter
                else:
                    self.chains[:, ch, iter] = self.chains[:, ch, iter-1]



    def MH_vectorized_sampling(self):

        return 1


    def gaussian_proposal(self, x_old, sigma:float = 0.01):
        x_new = x_old + np.random.randn(self.Ndim) * sigma
        return x_new



def Gaussian_liklihood(x):
    sigma = 5
    mean = 0


