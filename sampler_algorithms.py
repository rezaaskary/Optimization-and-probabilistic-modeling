import numpy as np
import scipy as sc
from tqdm import tqdm
import matplotlib.pyplot as plt
# import emcee
# from emcee import EnsembleSampler



class Metropolis_Hastings:
    def __init__(self,logprop_fcn, iterations:int = 1000, x0:np.ndarray = np.ones((1,1)), vectorized:bool = False, chains:int = 1):
        self.iterations = iterations
        self.x0 = x0
        self.Ndim = self.x0.shape[0]
        self.Nchain = chains
        self.chains = np.zeros((self.Ndim, self.Nchain, self.iterations))
        self.logprop_fcn = logprop_fcn
        self.logprop = np.zeros((self.Nchain, self.iterations))
        self.accept_rate = np.zeros((self.Nchain, self.iterations))
        self.chains[:, :, 0] = self.x0
        self.logprop[:,0] = self.logprop_fcn(self.x0)

    def MH_non_vectorized_sampling(self):
        uniform_random_number = np.random.uniform(low=0.0, high=1.0, size=(self.Nchain, self.iterations))
        for ch in tqdm(range(self.Nchain)):
            self.n_of_accept = 0
            for iter in range(1,self.iterations):
                self.proposed = self.gaussian_proposal(self.chains[:, :, iter-1], sigma = 0.05)


                Ln_prop = self.logprop_fcn(self.proposed)

                hastings = np.exp(Ln_prop) / np.exp(self.logprop[:,iter-1])
                min_ratio = hastings
                Index_min = hastings < 1
                min_ratio[Index_min] = hastings[Index_min]
                min_ratio[~Index_min] = 1
                criteria = uniform_random_number[ch,iter] < min_ratio
                if criteria:
                    self.chains[:, ch, iter:iter+1] = self.proposed

                    self.logprop[:, iter] = Ln_prop
                    self.n_of_accept += 1
                    self.accept_rate[ch,iter] = self.n_of_accept / iter
                else:
                    self.chains[:, ch, iter:iter+1] = self.chains[:, ch, iter - 1 : iter]
                    self.logprop[:, iter] = self.logprop[:, iter - 1]
        T1 = self.chains[0,0,:]
        T2 = self.chains[1, 0, :]
        plt.plot(T1)
        plt.plot(T2)
        plt.show()
        T
    #
    #
    # def MH_vectorized_sampling(self):
    #     return 1



    def gaussian_proposal(self, x_old, sigma:float = 0.01):
        x_old += np.random.randn(self.Ndim, 1) * sigma
        return x_old


def Gaussian_liklihood(parameter):
    sigma = parameter[0:1,0]
    x = parameter[1:,0]
    mean = 0
    log_gauss = -np.log(sigma * np.sqrt(2 * np.pi)) - ((x - mean) ** 2) / (2 * sigma ** 2)
    return log_gauss


 # logprop_fcn,
# logprop_fcn = Gaussian_liklihood,


if __name__=='__main__':
    x0 = 25 * np.ones((2, 1))
    x0[1,0] = 2
    G = Metropolis_Hastings(logprop_fcn = Gaussian_liklihood,iterations=100000, x0 = x0, vectorized=False,chains=1)
    G.MH_non_vectorized_sampling()