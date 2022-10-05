import numpy as np
from matplotlib.pyplot import plot, show
from IPython.display import display, Latex
#=========================================================================================================
class Continuous_Distributions:
    def __init__(self, sigma: float = None, variance: float = None,
                 mu: float = None, alpha: float = None,\
                 lb: float = None, ub: float = None, vectorized: bool = False,\
                 C: int = 1)->None:

        if isinstance(sigma, (float, int)) and isinstance(variance, (float, int)):
            raise Exception('Please Enter either variance or standard deviation!')

        if isinstance(sigma, (float, int)) and not isinstance(variance, (float, int)):
            if sigma > 0:
                self.sigma = sigma
                self.variance = sigma**2
            else:
                raise Exception('The standard deviation should be a positive value!')

        if not isinstance(sigma, (float, int)) and isinstance(variance, (float, int)):
            if variance > 0:
                self.sigma = np.sqrt(variance)
                self.variance = variance
            else:
                raise Exception('The standard deviation should be a positive value!')

        if sigma is None and variance is None:
            self.sigma = None
            self.variance = None

        if isinstance(lb, (float, int)):
            self.lb = lb
        elif lb is None:
            self.lb = None
        else:
            raise Exception('The lower bound is not specified correctly!')

        if isinstance(ub, (float, int)):
            self.ub = ub
        elif ub is None:
            self.ub = None
        else:
            raise Exception('The upper bound is not specified correctly!')

        if isinstance(vectorized, bool):
            self.vectorized = vectorized
        elif self.vectorized is None:
            self.vectorized = False
        else:
            raise Exception('The type of calculation is not specified correctly!')

        if isinstance(C, int):
            self.C = C
        elif C is None:
            self.C = 1
        else:
            raise Exception(' The number of chains is not specified correctly!')

        if isinstance(mu, (float, int)):
            self.mu = mu
        elif mu is None:
            self.mu = None
        else:
            raise Exception('The value of mu is not specified correctly!')

        if isinstance(alpha, (float, int)):
            self.alpha = alpha
        elif alpha is None:
            self.alpha = None
        else:
            raise Exception('The value of alpha is not specified correctly!')


class Uniform(Continuous_Distributions):
    def __init__(self, lb: float = None, ub: float = None, vectorized: bool = False, C: int = 1) -> None:
        super().__init__(lb, ub, vectorized, C)
        """
        The continuous uniform distribution
        :param lb: the lower bound of the uniform distribution
        :param ub: the upper bound of the uniform distribution
        :param vectorized: the type of calculating probablity distributions
        :param C: Number of chains
        """
        if self.vectorized:
            self.lb_v = self.lb * np.ones((self.C, 1))
            self.ub_v = self.ub * np.ones((self.C, 1))
            self.pdf = self.Prob_vectorized
            self.logpdf = self.Log_prob_vectorized
        else:
            self.pdf = self.Prob
            self.logpdf = self.Log_prob


    def Prob(self, x: float)->np.ndarray:
        """
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: the probablity of the occurance of the given variable
        """
        if x <= self.lb or x >= self.ub:
            return 0
        else:
            return 1 / (self.ub - self.lb)

    def Prob_vectorized(self, x:np.ndarray)->np.ndarray:
        """
        calculating the probability of the input array x in vectorized format
        :param x: the array of the input variable (Cx1)
        :return:  the probability of the input array (Cx1)
        """
        in_range_index = x>self.lb_v & x< self.ub_v
        prob = np.zeros((self.C,1))
        prob[in_range_index, 0] = 1/(self.ub_v[in_range_index, 0]- self.lb_v[in_range_index,0])
        return prob

    def Log_prob(self, x: float)->np.ndarray:
        """
        :param x: an integer value determining the variable we are calculating its probability distribution
        :return: The log of the probability distribution of the given variable
        """
        if x <= self.lb or x >= self.ub:
            return -np.inf
        else:
            return -np.log(self.ub - self.lb)

    def Log_prob_vectorized(self, x: np.ndarray)->np.ndarray:
        """
        calculating the log probability of the input array
        :param x: an array determining the variable we are calculating its probability distribution
        :return: The log of the probability distribution of the given variable
        """
        in_range_index = x > self.lb_v & x < self.ub_v
        logprob = np.ones((self.C, 1))
        logprob[in_range_index, 0] = -np.log(self.ub_v[in_range_index, 0] - self.lb_v[in_range_index, 0])
        return logprob

    def Visualize(self, lower_lim: float = -10, upper_lim: float = -10)->np.ndarray:
        """
        the module used to visualize the probablity distribution
        :param lower_lim: the lower limit used in ploting the probablity distribution
        :param upper_lim: the uppwer limit used in ploting the probablity distribution
        :return:
        """
        X = np.linspace(lower_lim, upper_lim, 1000)
        Y = list()
        for i in range(len(X)):
            Y.append(self.Prob(X[i]))
        plot(list(X.ravel()), Y)
        show()


#=========================================================================================================


class Gaussian(Continuous_Distributions):
    def __init__(self, sigma: float = None, variance: float = None, mu: float = None, vectorized: bool = False, C: int = 1) -> None:
        super().__init__(sigma, variance, mu, vectorized, C)
        """
        The continuous gaussian distribution function
        :param mu: the center of the gaussian distribution
        :param std: the standard deviation of gaussian distribution
        :param vectorized: the type of calculating probablity distributions
        :param C: Number of chains
        """
        if self.vectorized:
            self.mu_v = self.mu * np.ones((self.C, 1))
            self.sigma_v = self.sigma * np.ones((self.C, 1))
            self.pdf = self.Prob_vectorized
            self.logpdf = self.Log_prob_vectorized
        else:
            self.pdf = self.Prob
            self.logpdf = self.Log_prob

    def Prob(self, x: float)->np.ndarray:
        """
        calculating the probablity distribution of variable x by using normal distribution
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: the probablity of the occurance of the given variable
        """
        return (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2))

    def Log_prob(self, x: float)->np.ndarray:
        """
        calculating the log probablity distribution of variable x by using normal distribution
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: the probablity of the occurance of the given variable
        """
        return -np.log(self.sigma * np.sqrt(2 * np.pi)) - ((x - self.mu) ** 2) / (2 * self.sigma ** 2)

    def Prob_vectorized(self, x: np.ndarray)->np.ndarray:
       """
       calculating the probablity distribution of a chain variable x by using normal distribution
       :param x: an integer value determining the variable we are calculating its probablity distribution
       :return: the probablity of the occurance of the given variable
       """
       return (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2))

    def Log_prob_vectorized(self, x: np.ndarray)->np.ndarray:
        """
        calculating the log probablity distribution of the array of variable x by using normal distribution
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: the probablity of the occurance of the given array
        """
        return -np.log(self.sigma * np.sqrt(2 * np.pi)) - ((x - self.mu) ** 2) / (2 * self.sigma ** 2)

    def Visualize(self, lower_lim: float = -10, upper_lim: float = -10)->np.ndarray:
        """
         the module used to visualize the probablity distribution
         :param lower_lim: the lower limit used in ploting the probablity distribution
         :param upper_lim: the uppwer limit used in ploting the probablity distribution
         :return:
         """
        X = np.linspace(lower_lim, upper_lim, 1000)
        Y = list()
        for i in range(len(X)):
            Y.append(self.Prob(X[i]))
        plot(list(X.ravel()), Y)



class Truncated_Gaussian(Continuous_Distributions):
    def __init__(self, lb: float = None, ub: float = None, sigma: float = None, variance: float = None, mu: float = None, vectorized: bool = False, C: int = 1) -> None:
        super().__init__(lb, ub, sigma, variance, mu, vectorized, C)
        """
        The continuous truncated gaussian distribution function
        :param lb: the lower bound of the uniform distribution
        :param ub: the upper bound of the uniform distribution
        :param mu: the center of the gaussian distribution
        :param sigma: the standard deviation of gaussian distribution
        :param variance: the variance of gaussian distribution
        :param vectorized: the type of calculating probablity distributions
        :param C: Number of chains
        """

        if self.vectorized:
            self.lb_v = self.lb * np.ones((self.C, 1))
            self.ub_v = self.ub * np.ones((self.C, 1))
            self.mu_v = self.mu * np.ones((self.C, 1))
            self.sigma_v = self.sigma * np.ones((self.C, 1))
            self.pdf = self.Prob_vectorized
            self.logpdf = self.Log_prob_vectorized
        else:
            self.pdf = self.Prob
            self.logpdf = self.Log_prob


    def Erf(self, z)->np.ndarray:
        """
        The error function used to calculate the truncated gaussian distribution
        :param z: normalized input variable
        :return: the value of the error function
        """
        return (2 / (np.sqrt(np.pi))) * (z - (z ** 3 / 3) + (z ** 5 / 10) - (z ** 7 / 42) + (z ** 9 / 216))


    def Prob(self, x: float)->np.ndarray:
        """
        calcualting the probablity distribution of the truncated normal function
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: the probablity of the occurance of the given variable
        """
        if x <= self.lb or x >= self.ub:
            return 0
        else:
            L1 = (self.ub - self.mu) / self.sigma
            L2 = (self.lb - self.mu) / self.sigma
            L = (x - self.mu) / self.sigma
            Fi_1 = 0.5 * (1 + self.Erf(L1 / 2 ** 0.5))
            Fi_2 = 0.5 * (1 + self.Erf(L2 / 2 ** 0.5))
            fi = (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * L ** 2)
        return (1 / self.sigma) * (fi / (Fi_1 - Fi_2))


    def Log_prob(self, x: float)->np.ndarray:
        """
        calculating the log probablity of the truncated normal distribution
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: The log of the probablity distribution of the given variable
        """
        if x <= self.lb or x >= self.ub:
            return -np.inf
        else:
            L1 = (self.ub - self.mu) / self.std
            L2 = (self.lb - self.mu) / self.std
            L = (x - self.mu) / self.std
            Fi_1 = 0.5 * (1 + self.Erf(L1 / 2 ** 0.5))
            Fi_2 = 0.5 * (1 + self.Erf(L2 / 2 ** 0.5))
            return -np.log(self.std) - np.log(Fi_1 - Fi_2) - np.log((np.sqrt(2 * np.pi))) - 0.5 * L ** 2

    def Prob_vectorized(self, x: np.ndarray) -> np.ndarray:
        """
        calculating the probablity distribution of a chain variable x by using truncated normal distribution
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: the probablity of the occurance of the given variable
        """

        in_range_index = x > self.lb_v & x < self.ub_v
        prob = np.zeros((self.C, 1))
        L1 = (self.ub_v[in_range_index, 0] - self.mu_v[in_range_index, 0]) / self.sigma_v[in_range_index, 0]
        L2 = (self.lb_v[in_range_index, 0] - self.mu_v[in_range_index, 0]) / self.sigma_v[in_range_index, 0]
        L = (x[in_range_index, 0] - self.mu[in_range_index, 0]) / self.sigma[in_range_index, 0]
        Fi_1 = 0.5 * (1 + self.Erf(L1 / 2 ** 0.5))
        Fi_2 = 0.5 * (1 + self.Erf(L2 / 2 ** 0.5))
        fi = (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * L ** 2)
        prob[in_range_index, 0] = (1 / self.sigma[in_range_index, 0]) * (fi / (Fi_1 - Fi_2))
        return prob

    def Log_prob_vectorized(self, x: np.ndarray) -> np.ndarray:
        """
        calculating the log probablity of the truncated normal distribution(vectorized)
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: The log of the probablity distribution of the given variable
        """

        in_range_index = x > self.lb_v & x < self.ub_v
        logprob = np.ones((self.C, 1)) * -np.inf
        L1 = (self.ub_v[in_range_index, 0] - self.mu_v[in_range_index, 0]) / self.sigma_v[in_range_index, 0]
        L2 = (self.lb_v[in_range_index, 0] - self.mu_v[in_range_index, 0]) / self.sigma_v[in_range_index, 0]
        L = (x[in_range_index, 0] - self.mu[in_range_index, 0]) / self.sigma[in_range_index, 0]
        Fi_1 = 0.5 * (1 + self.Erf(L1 / 2 ** 0.5))
        Fi_2 = 0.5 * (1 + self.Erf(L2 / 2 ** 0.5))
        logprob[in_range_index, 0] = -np.log(self.std) - np.log(Fi_1 - Fi_2) - np.log((np.sqrt(2 * np.pi))) - 0.5 * L ** 2
        return logprob

    def Visualize(self, lower_lim: float = -10, upper_lim: float = -10):
        """
        the module used to visualize the probablity distribution
        :param lower_lim: the lower limit used in ploting the probablity distribution
        :param upper_lim: the uppwer limit used in ploting the probablity distribution
        :return:
        """
        X = np.linspace(lower_lim, upper_lim, 1000)
        Y = list()
        for i in range(len(X)):
            Y.append(self.Prob(X[i]))
        plot(list(X.ravel()), Y)




class Continuous_Half_Gaussian:
    def __init__(self, std: float = 1.0):
        """
        The half normal distribution function
        :param std: the standard deviation of the half normal distribution
        """
        self.std = std

    def Prob(self, x: float = 0.5)->np.ndarray:
        """
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: the probablity of the occurance of the given variable
        """
        if x <= 0:
            return 0
        else:
            return (np.sqrt(2) / (self.std * np.sqrt(np.pi))) * np.exp(-((x - self.mu) ** 2) / (2 * self.std ** 2))

    def Log_prob(self, x: float = 0.5)->np.ndarray:
        """
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: The log of the probablity distribution of the given variable
        """
        if x <= 0:
            return -np.inf
        else:
            return np.log(np.sqrt(2) / (self.std * np.sqrt(np.pi))) - (x ** 2) / (2 * self.std ** 2)

    def Visualize(self, lower_lim: float = -10, upper_lim: float = -10):
        """
        the module used to visualize the probablity distribution
        :param lower_lim: the lower limit used in ploting the probablity distribution
        :param upper_lim: the uppwer limit used in ploting the probablity distribution
        :return:
        """
        X = np.linspace(lower_lim, upper_lim, 1000)
        Y = list()
        for i in range(len(X)):
            Y.append(self.Prob(X[i]))
        plot(list(X.ravel()), Y)



class Skewed_Normal:
    def __init__(self, mu: float = 0.0, std: float = 1.0, alpha: float = 1):
        """
       The continuous skewed Normal distribution
       :param mu: the center bound of the truncated normal distribution
       :param std: the standard deviation bound of the truncated normal distribution
         :param alpha: the skewness parameter
        """
        self.mu = mu
        self.std = std
        self.alpha = alpha
    def Erf(self, z)->np.ndarray:
        """
        The error function used to calculate the truncated gaussian distribution
        :param z: normalized input variable
        :return: the value of the error function
        """
        return (2 / (np.sqrt(np.pi))) * (z - (z ** 3 / 3) + (z ** 5 / 10) - (z ** 7 / 42) + (z ** 9 / 216))

    def Prob(self, x)->np.ndarray:
        """
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: the probablity of the occurance of the given variable
        """
        L1 = 0.5 * (1 + self.Erf(((x -self.mu)/self.std)*(self.alpha/np.sqrt(2.0))))
        L2 = (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x -self.mu)/self.std)**2)
        return 2 * L1 * L2

    def Log_prob(self, x: float = 0.5)->np.ndarray:
        """
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: The log of the probablity distribution of the given variable
        """
        return np.log(0.5 * (1 + self.Erf(((x -self.mu)/self.std)*(self.alpha/np.sqrt(2.0))))) - np.log((np.sqrt(2 * np.pi))) - 0.5 * ((x -self.mu)/self.std)**2



    def Visualize(self, lower_lim: float = -10, upper_lim: float = -10):
        """
        the module used to visualize the probablity distribution
        :param lower_lim: the lower limit used in ploting the probablity distribution
        :param upper_lim: the uppwer limit used in ploting the probablity distribution
        :return:
        """
        X = np.linspace(lower_lim, upper_lim, 1000)
        Y = list()
        for i in range(len(X)):
            Y.append(self.Prob(X[i]))
        plot(list(X.ravel()), Y)