import numpy as np
from matplotlib.pyplot import plot, show
from mathmatics import Beta, Gamma, Erf
#=========================================================================================================
class Continuous_Distributions:
    def __init__(self, sigma: float = None, variance: float = None,
                 mu: float = None, alpha: float = None,\
                 lb: float = None, ub: float = None, vectorized: bool = False,\
                 C: int = 1, beta: float = None)->None:

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

        if isinstance(beta, (float, int)):
            self.beta = beta
        elif beta is None:
            self.beta = None
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

        self.pdf = self.Prob
        self.logpdf = self.Log_prob
        self.cdf = self.CDF

    def Prob(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        calculating the probability of the input array x in vectorized format
        :param x: the array of the input variable (Cx1)
        :return:  the probability of the input array (Cx1) and the derivatives of the probablity distribution (Cx1)
        """
        in_range_index = (x>self.lb) & (x< self.ub)
        prob = np.zeros((self.C, 1))
        prob[in_range_index[:,0], 0] = 1/(self.ub- self.lb)
        derivatives_prob = np.zeros((self.C, 1))
        return prob, derivatives_prob

    def Log_prob(self, x: np.ndarray)->(np.ndarray, np.ndarray):
        """
        calculating the log probability of the input array
        :param x: an array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log (and the derivatives log) of the probability distribution of the given variable (Cx1)
        """
        in_range_index = (x > self.lb) & (x < self.ub)
        logprob = np.ones((self.C, 1)) * (-np.inf)
        derivatives_logprob = logprob.copy()
        logprob[in_range_index[:,0], 0] = -np.log(self.ub - self.lb)
        derivatives_logprob[in_range_index[:,0], 0] = 0
        return logprob, derivatives_logprob

    def CDF(self, x: np.ndarray)->np.ndarray:
        """
        calculating the CDF probability of the input array
        :param x: an array determining the variable we are calculating its probability distribution
        :return: The log of the probability distribution of the given variable
        """
        left_index = x <= self.lb
        right_index = x >= self.lb
        in_range_index = (x > self.lb) & (x < self.ub)
        cdf = np.ones((self.C, 1))
        cdf[left_index[:,0], 0] = 0
        cdf[right_index[:, 0], 0] = 1
        cdf[in_range_index[:, 0], 0] = (x[in_range_index[:, 0], 0] - self.lb)/(self.ub - self.lb)
        return cdf

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

class Normal(Continuous_Distributions):
    def __init__(self, sigma: float = None, variance: float = None, mu: float = None, vectorized: bool = False, C: int = 1) -> None:
        super().__init__(sigma, variance, mu, vectorized, C)
        """
        The continuous gaussian distribution function
        :param mu: the center of the gaussian distribution
        :param std: the standard deviation of gaussian distribution
        :param vectorized: the type of calculating probablity distributions
        :param C: Number of chains
        """
        self.Erf = Erf
        self.pdf = self.Prob
        self.logpdf = self.Log_prob
        self.cdf = self.CDF

    def Prob(self, x: np.ndarray)->(np.ndarray, np.ndarray):
        """
        calculating the probablity distribution of variable x by using normal distribution (Cx1)
        :param x: an integer value determining the variable we are calculating its probablity distribution (Cx1)
        :return: the probablity of the occurance of the given variable (Cx1)
        """
        prob = (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2))
        derivatives_prob = (-1 / (self.sigma**3)) * np.sqrt(2/np.pi) * (x - self.mu) * np.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2))
        return prob, derivatives_prob

    def Log_prob(self, x: float)->(np.ndarray, np.ndarray):
        """
        calculating the log probablity  and the derivatives of distribution of variable x by using normal distribution (Cx1)
        :param x: an integer value determining the variable we are calculating its probablity distribution (Cx1)
        :return: the log probablity and the derivatives of the occurance of the given variable (Cx1)
        """
        log_prob = -np.log(self.sigma * np.sqrt(2 * np.pi)) - ((x - self.mu) ** 2) / (2 * self.sigma ** 2)
        derivatives_log_prob = -(x - self.mu)/(self.sigma ** 2)
        return log_prob, derivatives_log_prob

    def CDF(self, x: np.ndarray)->(np.ndarray, np.ndarray):
        """
        calculating the CDF probability of the input array
        :param x: an array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log of the probability distribution of the given variable (Cx1)
        """
        z = (x-self.mu)/(self.sigma * np.sqrt(2))
        erf_value, derivatives_value = self.Erf(z)
        return erf_value, derivatives_value/(self.sigma * np.sqrt(2))

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


class Truncated_Normal(Continuous_Distributions):
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
        self.Erf = Erf
        self.pdf = self.Prob
        self.logpdf = self.Log_prob
        self.cdf = self.CDF

    def Prob(self, x: np.ndarray)->(np.ndarray, np.ndarray):
        """
        calcualting the probablity distribution of the truncated normal function
        :param x: an integer value determining the variable we are calculating its probablity distribution (Cx1)
        :return: the probablity (and the derivatives) of the occurance of the given variable (Cx1)
        """
        in_range_index = (x >= self.lb) & (x <= self.ub)
        prob = np.zeros((self.C, 1))
        der_prob = np.zeros((self.C, 1))

        arg_r = (self.ub - self.mu) / self.sigma
        arg_l = (self.lb - self.mu) / self.sigma

        erf_r = 0.5 * (1 + self.Erf(arg_r / np.sqrt(2)))
        ert_l = 0.5 * (1 + self.Erf(arg_l / np.sqrt(2)))

        normal_argument = (x[in_range_index[:,0],0] - self.mu) / self.sigma
        normal_fcn_value = (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * normal_argument ** 2)

        prob[in_range_index[:,0], 0] =  (1 / self.sigma) * (normal_fcn_value / (erf_r - ert_l))
        der_prob[in_range_index[:,0], 0] = (1 / self.sigma**2) * (1/(erf_r - ert_l)) * (-1 / (np.sqrt(2 * np.pi))) * normal_argument * np.exp(-0.5 * normal_argument ** 2)

        return prob, der_prob


    def Log_prob(self, x: np.ndarray)->(np.ndarray, np.ndarray):
        """
        calculating the log probablity of the truncated normal distribution
        :param x: an integer value determining the variable we are calculating its probablity distribution (Cx1)
        :return: The log (and its derivatives) of the probablity distribution of the given variable (Cx1)
        """

        in_range_index = (x >= self.lb) & (x <= self.ub)
        logprob = np.ones((self.C, 1)) * -np.inf
        derivatives_logprob = np.ones((self.C, 1)) * -np.inf
        arg_r = (self.ub - self.mu) / self.sigma
        arg_l = (self.lb - self.mu) / self.sigma
        normal_argument = (x[in_range_index[:, 0], 0] - self.mu) / self.sigma

        erf_r = 0.5 * (1 + self.Erf(arg_r / np.sqrt(2)))
        ert_l = 0.5 * (1 + self.Erf(arg_l / np.sqrt(2)))

        logprob[in_range_index[:, 0], 0] =  -np.log(self.sigma) - np.log(erf_r - ert_l) - 0.5 * np.log(2 * np.pi) - 0.5 * normal_argument ** 2
        derivatives_logprob [in_range_index[:, 0], 0] = (-1 / self.sigma**2) * (x[in_range_index[:, 0], 0] - self.mu)
        return logprob, derivatives_logprob


    def CDF(self, x: np.ndarray)->(np.ndarray, np.ndarray):
        """
        calculating the CDF probability of the input array
        :param x: an array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log of the probability distribution of the given variable (Cx1)
        """

        right_index = x > self.ub
        in_range_index = (x >= self.lb) & (x <= self.ub)
        cdf = np.zeros((self.C, 1))
        cdf[right_index[:, 0], 0] = 1.0

        b = (self.ub - self.mu) / self.sigma
        a = (self.lb - self.mu) / self.sigma
        xi = (x[in_range_index[:, 0], 0] - self.mu) / self.sigma

        erf_r = 0.5 * (1 + self.Erf( b / np.sqrt(2)))
        ert_l = 0.5 * (1 + self.Erf( a / np.sqrt(2)))
        ert_xi = 0.5 * (1 + self.Erf( xi / np.sqrt(2)))
        cdf[in_range_index[:, 0], 0] = (ert_xi - ert_l) / (erf_r - ert_l)
        derivatives_value = None
        return cdf, derivatives_value


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


class Half_Normal(Continuous_Distributions):
    def __init__(self, sigma: float = None, variance: float = None, vectorized: bool = False, C: int = 1) -> None:
        super().__init__(sigma, variance, vectorized, C)
        """
        The continuous truncated gaussian distribution function
        :param sigma: the standard deviation of gaussian distribution
        :param variance: the variance of gaussian distribution
        :param vectorized: the type of calculating probablity distributions
        :param C: Number of chains
        """

        self.Erf = Erf
        self.pdf = self.Prob
        self.logpdf = self.Log_prob
        self.cdf = self.CDF


    def Prob(self, x: np.ndarray)->(np.ndarray, np.ndarray):
        """
        Calculating the probablity of half normal distribution
        :param x: an array value determining the variable we are calculating its probablity distribution (Cx1)
        :return: the probablity and the derivatives of the occurance of the given variable (Cx1)
        """
        in_range_index = (x >= 0)
        prob = np.zeros((self.C, 1))
        derivatives_prob = np.zeros((self.C, 1))
        prob[in_range_index[:, 0], 0] = (np.sqrt(2 / np.pi) / self.sigma) * np.exp(-((x[in_range_index[:, 0], 0]) ** 2) / (2 * self.sigma ** 2))
        derivatives_prob[in_range_index[:, 0], 0] = (- np.sqrt(2 / np.pi) / (self.sigma ** 3)) * (x[in_range_index[:, 0], 0]) * np.exp(-((x[in_range_index[:, 0], 0]) ** 2) / (2 * self.sigma ** 2))

        return prob, derivatives_prob


    def Log_prob(self, x: np.ndarray)->np.ndarray:
        """
        Calcualting the log probablity of the half normal distribution
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: The log of the probablity distribution of the given variable
        """

        in_range_index = (x >= 0)
        logprob = np.ones((self.C, 1)) * -np.inf
        derivatives_logprob = np.ones((self.C, 1)) * -np.inf
        logprob[in_range_index[:, 0], 0] = 0.5 * np.log(2/np.pi) - np.log(self.sigma) - (x[in_range_index[:, 0], 0]** 2) / (2 * self.std ** 2)
        derivatives_logprob[in_range_index[:, 0], 0] = -x[in_range_index[:, 0], 0] / self.sigma**2
        return logprob, derivatives_logprob

    def CDF(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        calculating the CDF probability of the input array
        :param x: an array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log of the probability distribution of the given variable (Cx1)
        """
        in_range_index = (x >= 0)
        cdf = np.zeros((self.C, 1))
        derivatives_value =  np.zeros((self.C, 1))
        z = x[in_range_index[:, 0], 0] / (self.sigma * np.sqrt(2))
        erf_value, derivatives_erf = self.Erf(x[in_range_index[:, 0], 0] / (self.sigma * np.sqrt(2)))
        derivatives_value[in_range_index[:, 0], 0] = derivatives_erf * (1/(self.sigma * np.sqrt(2)))
        cdf[in_range_index[:, 0], 0] = erf_value
        return cdf, derivatives_value


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


class Skewed_Normal(Continuous_Distributions):
    def __int__(self, mu: float = None , alpha: float = None, sigma: float = None, variance: float = None, vectorized: bool = False, C: int = 1)->None:
        super().__int__(mu, alpha, sigma, variance, vectorized, C)
        """
        The skewed continuous truncated gaussian distribution function
        :param alpha: the skewness parameter
        :param mu: the mean of the gaussian distribution 
        :param sigma: the standard deviation of gaussian distribution
        :param variance: the variance of gaussian distribution
        :param vectorized: the type of calculating probablity distributions
        :param C: Number of chains
        """
        self.Erf = Erf
        self.pdf = self.Prob
        self.logpdf = self.Log_prob
        self.cdf = self.CDF


    def Prob(self, x)->np.ndarray:
        """
        calculating the probablity distribution of the skewed normal function
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: the probablity of the occurance of the given variable
        """
        z = (x - self.mu) / self.sigma
        erf_part, der_erf_part = 0.5 * (1 + self.Erf(z * (self.alpha / np.sqrt(2.0))))
        normal_part = (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * (z) ** 2)
        prob = 2 * erf_part * normal_part
        derivatives_prob = -np.sqrt(2/np.pi) * (z/self.sigma) * np.exp(-0.5 * (z) ** 2) * erf_part + (self.alpha/self.sigma) * np.sqrt(2/np.pi) * np.exp(-0.5 * (z) ** 2) * der_erf_part
        return prob, derivatives_prob


    def Log_prob(self, x: float)->np.ndarray:
        """
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: The log of the probablity distribution of the given variable
        """
        z = (x - self.mu) / self.sigma

        erf_value, der_erf_value = self.Erf((z * self.alpha)/np.sqrt(2))

        log_prob = -0.5 * np.log(2 * np.pi) - 0.5 * (z**2) + np.log(1 + erf_value)
        derivatives_log_prob = -z * (1 / self.sigma) + (1 / (self.sigma * np.sqrt(2))) * (der_erf_value / (1 + erf_value))
        return log_prob, derivatives_log_prob


    def Prob_vectorized(self, x:np.ndarray)->np.ndarray:
        """
        calcualting the probablity of the skewed normal distribution given an array of input variable
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: The log of the probablity distribution of the given variable
        """
        L1 = 0.5 * (1 + self.Erf(((x - self.mu_v) / self.sigma_v) * (self.alpha_v / np.sqrt(2.0))))
        L2 = (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - self.mu_v) / self.sigma_v) ** 2)
        return 2 * L1 * L2

    def Log_prob_vectorized(self, x:np.ndarray)->np.ndarray:
        """
        calculates the log probablity of the skewed normal distribution given an array on inputs
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: The log of the probablity distribution of the given variable
        """
        log_prob = np.log(0.5 * (1 + self.Erf(((x - self.mu_v) / self.sigma_v) * (self.alpha_v / np.sqrt(2.0))))) - np.log(
            (np.sqrt(2 * np.pi))) - 0.5 * ((x - self.mu_v) / self.sigma_v) ** 2
        return log_prob

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



class Beta(Continuous_Distributions):
    def __int__(self, alpha: None, beta: None, vectorized: bool = False, C: int = 1)->None:
        super().__int__(alpha, beta, vectorized, C)
        """
        Initializing beta distribution continuous function
        :param alpha: exponent alpha parameter (alpha>0)
        :param beta:  exponent beta parameter (beta>0)
        :param vectorized: boolean variable used to determine vectorized calculation
        :param C: An integer variable indicating the number of chains 
        :return: None
        """

        if self.alpha <= 0:
            raise Exception('Parameter alpha (for calculating the beta distribution) should be positive')
        if self.beta <= 0:
            raise Exception('Parameter beta (for calculating the beta distribution) should be positive')

        self.mean = (self.alpha)/(self.alpha + self.beta)
        self.variance = (self.alpha * self.beta) / (((self.alpha + self.beta)**2) * (self.alpha + self.beta + 1))


        self.Beta = Beta
        if self.vectorized:
            self.alpha_v = self.alpha * np.ones((self.C, 1))
            self.beta_v = self.beta * np.ones((self.C, 1))
            self.pdf = self.Prob_vectorized
            self.logpdf = self.Log_prob_vectorized
        else:
            self.pdf = self.Prob
            self.logpdf = self.Log_prob


    def Prob(self, x: float)->float:
        """
        calculating the probablity distribution of the Beta distribution
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: the probablity of the occurance of the given variable
        """
        x = np.clip(x, 0, 1)
        return ((x**(self.alpha - 1)) * ((1 - x)**(self.beta - 1))) / self.Beta(self.alpha, self.beta)

    def Log_prob(self, x: float)->float:
        """
        calculating the log probablity distribution of the Beta distribution
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: the probablity of the occurance of the given variable
        """
        x = np.clip(x, 0, 1)
        return (self.alpha - 1) * np.log(x) + (self.beta - 1) * np.log(1 - x) - np.log(self.Beta(self.alpha, self.beta))

    def Prob_vectorized(self,x: np.ndarray)->np.ndarray:
        """
        calculating the probablity distribution of the Beta distribution given an array of the input variables
        :param x: an array value determining the variable we are calculating its probablity distribution (cx1)
        :return: the probablity of the occurance of the given variable
        """
        x = np.clip(x, 0, 1)
        return ((x ** (self.alpha - 1)) * ((1 - x) ** (self.beta - 1))) / self.Beta(self.alpha, self.beta)

    def Log_prob_vectorized(self,x: np.ndarray)->np.ndarray:
        """
        calculating the log probablity distribution of the Beta distribution given any array on the input
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: the probablity of the occurance of the given variable
        """
        x = np.clip(x, 0, 1)
        return (self.alpha - 1) * np.log(x) + (self.beta - 1) * np.log(1 - x) - np.log(self.Beta(self.alpha, self.beta))


class Kumaraswamy(Continuous_Distributions):
    def __int__(self, alpha: None, beta: None, vectorized: bool = False, C: int = 1)->None:
        super().__int__(alpha, beta, vectorized, C)
        """
        Initializing Kumaraswamy distribution continuous function
        :param alpha: exponent alpha parameter (alpha>0)
        :param beta:  exponent beta parameter (beta>0)
        :param vectorized: boolean variable used to determine vectorized calculation
        :param C: An integer variable indicating the number of chains 
        :return: None
        """

        if self.alpha <= 0:
            raise Exception('Parameter alpha (for calculating the beta distribution) should be positive')
        if self.beta <= 0:
            raise Exception('Parameter beta (for calculating the beta distribution) should be positive')

        self.mean = (self.beta * Gamma(1 + 1/self.alpha) * Gamma(self.beta))/( Gamma(self.beta + 1 + 1/self.alpha))

        if self.vectorized:
            self.alpha_v = self.alpha * np.ones((self.C, 1))
            self.beta_v = self.beta * np.ones((self.C, 1))
            self.pdf = self.Prob_vectorized
            self.logpdf = self.Log_prob_vectorized
        else:
            self.pdf = self.Prob
            self.logpdf = self.Log_prob


    def Prob(self, x: float)->float:
        """
        calculating the probablity distribution of the Kumaraswamy distribution
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: the probablity of the occurance of the given variable
        """
        x = np.clip(x, 0, 1)
        return self.beta * self.alpha * (x** (self.alpha-1)) * ((1 - x**self.alpha)**(self.beta - 1))

    def Log_prob(self, x: float)->float:
        """
        calculating the log probablity distribution of the Kumaraswamy distribution
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: the probablity of the occurance of the given variable
        """
        x = np.clip(x, 0, 1)
        return np.log(self.alpha) + np.log(self.beta) + (self.alpha - 1) * np.log(x) + (self.beta - 1) * np.log((1 - x**self.alpha))

    def Prob_vectorized(self,x: np.ndarray)->np.ndarray:
        """
        calculating the probablity distribution of the Kumaraswamy distribution given an array of the input variables
        :param x: an array value determining the variable we are calculating its probablity distribution (cx1)
        :return: the probablity of the occurance of the given variable
        """
        x = np.clip(x, 0, 1)
        return self.beta * self.alpha * (x** (self.alpha-1)) * ((1 - x**self.alpha)**(self.beta - 1))

    def Log_prob_vectorized(self,x: np.ndarray)->np.ndarray:
        """
        calculating the log probablity distribution of the Kumaraswamy distribution given any array on the input
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: the probablity of the occurance of the given variable
        """
        x = np.clip(x, 0, 1)
        return np.log(self.alpha) + np.log(self.beta) + (self.alpha - 1) * np.log(x) + (self.beta - 1) * np.log((1 - x**self.alpha))