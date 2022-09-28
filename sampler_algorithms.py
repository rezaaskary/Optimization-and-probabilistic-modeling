import numpy as np
import scipy as sc
from tqdm import tqdm
import matplotlib.pyplot as plt
# import emcee
# from emcee import EnsembleSampler


def Gaussian_liklihood(parameter):
    x = parameter[0:1,0]
    # x = parameter[1:,0]
    mean = 0
    sigma = 2
    log_gauss = -np.log(sigma * np.sqrt(2 * np.pi)) - ((x - mean) ** 2) / (2 * sigma ** 2)
    return log_gauss



class Metropolis_Hastings:
    def __init__(self,logprop_fcn, iterations:int = 1000, x0:np.ndarray = np.ones((1,1)), vectorized:bool = False, chains:int = 1):
        # checking the correctness of the iteration
        if isinstance(iterations,int):
            self.iterations = iterations
        else:
            self.iterations = 1000
            print(r'The iteration is not a integer value. The default value of {self.iterations} is selectd as the number of iterations')











        self.x0 = x0
        self.Ndim = self.x0.shape[0]
        self.Nchain = chains
        self.chains = np.zeros((self.Ndim, self.Nchain, self.iterations))
        self.logprop_fcn = logprop_fcn
        self.logprop = np.zeros((self.Nchain, self.iterations))
        self.accept_rate = np.zeros((self.Nchain, self.iterations))
        self.chains[:, :, 0] = self.x0
        self.logprop[:,0] = self.logprop_fcn(self.x0)
        self.n_of_accept = np.zeros((self.Nchain, 1))


    def MH_non_vectorized_sampling(self):
        uniform_random_number = np.random.uniform(low=0.0, high=1.0, size=(self.Nchain, self.iterations))
        for iter in tqdm(range(1, self.iterations)):
            for ch in (range(self.Nchain)):
                # generating the sample for each chain
                self.proposed = self.gaussian_proposal(self.chains[:, ch, iter-1:iter].copy(), sigma = 0.1)
                # calculating the log of the posteriori function
                Ln_prop = self.logprop_fcn(self.proposed)
                # calculating the hasting ratio
                hastings = np.exp(Ln_prop - self.logprop[ch,iter-1])
                criteria = uniform_random_number[ch,iter]< hastings
                if criteria:
                    self.chains[:, ch, iter:iter+1] = self.proposed
                    self.logprop[ch, iter] = Ln_prop
                    self.n_of_accept += 1
                    self.accept_rate[ch,iter] = self.n_of_accept / iter
                else:
                    self.chains[:, ch, iter:iter+1] = self.chains[:, ch, iter - 1 : iter]
                    self.logprop[ch, iter] = self.logprop[ch, iter - 1]
                    self.accept_rate[ch, iter] = self.n_of_accept[ch] / iter
        T1 = self.chains[0,0,:]
        # T2 = self.chains[1, 0, :]
        plt.plot(T1)
        # plt.plot(T2)
        plt.show()
        plt.figure()
        plt.plot( self.accept_rate.ravel())
        plt.show()
        T
    #
    #
    def MH_vectorized_sampling(self):
        return 1



    def gaussian_proposal(self, x_old, sigma:float = 0.01):
        x_old += np.random.randn(self.Ndim, 1) * sigma
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
    G = Metropolis_Hastings(logprop_fcn = Gaussian_liklihood,iterations=500.1, x0 = x0, vectorized=False,chains=1)
    G.MH_non_vectorized_sampling()