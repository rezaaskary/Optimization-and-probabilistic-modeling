import numpy as np
from matplotlib.pyplot import plot, show
from mathmatics import Beta, Gamma, Erf
#=========================================================================================================
class Continuous_Distributions:
    def __init__(self, variance: float = None, sigma: float=None, mu: float = None,\
                lb: float = None, ub: float = None, alpha: float = None,\
                 a:float=None, b:float=None, vectorized: bool = True,\
                 C: int = 1, beta: float = None, Lambda:float = None,\
                 kappa:float = None, nu:float = None)->None:

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


        if isinstance(Lambda, (float, int)):
            self.Lambda = Lambda
        elif Lambda is None:
            self.Lambda = None
        else:
            raise Exception('The value of lambda is not specified correctly!')

        if isinstance(a, (float, int)):
            self.a = a
        elif a is None:
            self.a = None
        else:
            raise Exception('The value of a is not specified correctly!')

        if isinstance(b, (float, int)):
            self.b = b
        elif b is None:
            self.b = None
        else:
            raise Exception('The value of b is not specified correctly!')


        if isinstance(kappa, (float, int)):
            self.kappa = kappa
        elif kappa is None:
            self.kappa = None
        else:
            raise Exception('The value of kappa is not specified correctly!')

        if isinstance(nu, (float, int)):
            self.nu = nu
        elif nu is None:
            self.nu = None
        else:
            raise Exception('The value of nu is not specified correctly!')



    def Visualize(self, lower_lim: float = -10, upper_lim: float = -10):
        """
        Visualizing the probablity distribution
        :param lower_lim: the lower limit used in ploting the probablity distribution
        :param upper_lim: the uppwer limit used in ploting the probablity distribution
        :return: a line plot from matplotlib library
        """
        X = np.linspace(lower_lim, upper_lim, 1000)
        Y = list()
        for i in range(len(X)):
            Y.append(self.Prob(X[i]))
        plot(list(X.ravel()), Y)


class Uniform(Continuous_Distributions):
    def __init__(self, a: float = None, b: float = None, vectorized: bool = False, C: int = 1) -> None:
        super(Uniform,self).__init__(a = a, b = b, vectorized = vectorized, C = C)
        """
        The continuous uniform distribution
        :param lb: the lower bound of the uniform distribution
        :param ub: the upper bound of the uniform distribution
        :param vectorized: the type of calculating probablity distributions
        :param C: Number of chains
        """

        if self.a >= self.b:
            raise Exception('The lower limit of the uniform distribution is greater than the upper limit!')

        self.pdf = self.Prob
        self.logpdf = self.Log_prob
        self.cdf = self.CDF

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def Prob(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the probablity of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probablity distribution (Cx1)
        :return: The probablity (and the derivative) of the occurance of the given variable (Cx1, Cx1)
        """
        in_range_index = (x>self.a) & (x< self.b)
        prob = np.zeros((self.C, 1))
        prob[in_range_index[:,0], 0] = 1/(self.b- self.a)
        derivatives_prob = np.zeros((self.C, 1))
        return prob, derivatives_prob

    def Log_prob(self, x: np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the ---- distribution
        :param x: An integer array determining the variable we are calculating its probablity distribution (Cx1)
        :return: The log probablity and derivatives of the log probablity of the occurance of an independent variable (Cx1, Cx1)
        """
        in_range_index = (x > self.a) & (x < self.b)
        logprob = np.ones((self.C, 1)) * (-np.inf)
        derivatives_logprob = logprob.copy()
        logprob[in_range_index[:,0], 0] = -np.log(self.b - self.a)
        derivatives_logprob[in_range_index[:,0], 0] = 0
        return logprob, derivatives_logprob

    def CDF(self, x: np.ndarray)->np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its detivatives) with respect to the input variable (Cx1, Cx1)
        """
        left_index = x <= self.a
        right_index = x >= self.a
        in_range_index = (x > self.a) & (x < self.b)
        cdf = np.ones((self.C, 1))
        cdf[left_index[:,0], 0] = 0
        cdf[right_index[:, 0], 0] = 1
        cdf[in_range_index[:, 0], 0] = (x[in_range_index[:, 0], 0] - self.a)/(self.b - self.a)
        return cdf

#=========================================================================================================
class Normal(Continuous_Distributions):
    def __init__(self, sigma: float = None, variance: float = None, mu: float = None, vectorized: bool = False, C: int = 1) -> None:
        super(Normal,self).__init__(sigma = sigma, variance = variance, mu = mu, vectorized = vectorized, C = C)
        """
        The continuous gaussian distribution function
        :param mu: the center of the gaussian distribution
        :param std: the standard deviation of gaussian distribution
        :param vectorized: the type of calculating probablity distributions
        :param C: Number of chains
        """
        if self.mu is None or self.sigma is None:
            raise Exception('The value of either mean or standard deviation is not specified (Normal distribution)!')


        self.Erf = Erf
        self.pdf = self.Prob
        self.logpdf = self.Log_prob
        self.cdf = self.CDF

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def Prob(self, x: np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the probablity of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probablity distribution (Cx1)
        :return: The probablity (and the derivative) of the occurance of the given variable (Cx1, Cx1)
        """
        prob = (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2))
        derivatives_prob = (-1 / (self.sigma**3)) * np.sqrt(2/np.pi) * (x - self.mu) * np.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2))
        return prob, derivatives_prob

    def Log_prob(self, x: float)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the ---- distribution
        :param x: An integer array determining the variable we are calculating its probablity distribution (Cx1)
        :return: The log probablity and derivatives of the log probablity of the occurance of an independent variable (Cx1, Cx1)
        """
        log_prob = -np.log(self.sigma * np.sqrt(2 * np.pi)) - ((x - self.mu) ** 2) / (2 * self.sigma ** 2)
        derivatives_log_prob = -(x - self.mu)/(self.sigma ** 2)
        return log_prob, derivatives_log_prob

    def CDF(self, x: np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the cumulative distribution function for ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its detivatives) with respect to the input variable (Cx1, Cx1)
        """
        z = (x-self.mu)/(self.sigma * np.sqrt(2))
        erf_value, derivatives_value = self.Erf(z)
        return erf_value, derivatives_value/(self.sigma * np.sqrt(2))



class Truncated_Normal(Continuous_Distributions):
    def __init__(self, lb: float = None, ub: float = None, sigma: float = None, variance: float = None, mu: float = None, vectorized: bool = False, C: int = 1) -> None:
        super(Truncated_Normal,self).__init__(lb = lb, ub = ub, sigma = sigma, variance = variance, mu = mu, vectorized = vectorized, C = C)
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

        if self.lb >= self.ub:
            raise Exception('The lower limit of the truncated Normal distribution is greater than the upper limit!')
        if self.mu is None or self.sigma is None:
            raise Exception('The value of either mean or standard deviation is not specified (Truncated Normal distribution)!')

        self.Erf = Erf
        self.pdf = self.Prob
        self.logpdf = self.Log_prob
        self.cdf = self.CDF

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def Prob(self, x: np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the probablity of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probablity distribution (Cx1)
        :return: The probablity (and the derivative) of the occurance of the given variable (Cx1, Cx1)
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
        Parallelized calculating the log (and its derivatives) of the ---- distribution
        :param x: An integer array determining the variable we are calculating its probablity distribution (Cx1)
        :return: The log probablity and derivatives of the log probablity of the occurance of an independent variable (Cx1, Cx1)
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
        Parallelized calculating the cumulative distribution function for ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its detivatives) with respect to the input variable (Cx1, Cx1)
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




class Half_Normal(Continuous_Distributions):
    def __init__(self, sigma: float = None, variance: float = None, vectorized: bool = False, C: int = 1) -> None:
        super(Half_Normal,self).__init__(sigma = sigma, variance = variance, vectorized = vectorized, C = C)
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

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def Prob(self, x: np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the probablity of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probablity distribution (Cx1)
        :return: The probablity (and the derivative) of the occurance of the given variable (Cx1, Cx1)
        """
        in_range_index = (x >= 0)
        prob = np.zeros((self.C, 1))
        derivatives_prob = np.zeros((self.C, 1))
        prob[in_range_index[:, 0], 0] = (np.sqrt(2 / np.pi) / self.sigma) * np.exp(-((x[in_range_index[:, 0], 0]) ** 2) / (2 * self.sigma ** 2))
        derivatives_prob[in_range_index[:, 0], 0] = (- np.sqrt(2 / np.pi) / (self.sigma ** 3)) * (x[in_range_index[:, 0], 0]) * np.exp(-((x[in_range_index[:, 0], 0]) ** 2) / (2 * self.sigma ** 2))

        return prob, derivatives_prob


    def Log_prob(self, x: np.ndarray)->np.ndarray:
        """
        Parallelized calculating the log (and its derivatives) of the ---- distribution
        :param x: An integer array determining the variable we are calculating its probablity distribution (Cx1)
        :return: The log probablity and derivatives of the log probablity of the occurance of an independent variable (Cx1, Cx1)
        """

        in_range_index = (x >= 0)
        logprob = np.ones((self.C, 1)) * -np.inf
        derivatives_logprob = np.ones((self.C, 1)) * -np.inf
        logprob[in_range_index[:, 0], 0] = 0.5 * np.log(2/np.pi) - np.log(self.sigma) - (x[in_range_index[:, 0], 0]** 2) / (2 * self.std ** 2)
        derivatives_logprob[in_range_index[:, 0], 0] = -x[in_range_index[:, 0], 0] / self.sigma**2
        return logprob, derivatives_logprob

    def CDF(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the cumulative distribution function for ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its detivatives) with respect to the input variable (Cx1, Cx1)
        """
        in_range_index = (x >= 0)
        cdf = np.zeros((self.C, 1))
        derivatives_value =  np.zeros((self.C, 1))
        z = x[in_range_index[:, 0], 0] / (self.sigma * np.sqrt(2))
        erf_value, derivatives_erf = self.Erf(x[in_range_index[:, 0], 0] / (self.sigma * np.sqrt(2)))
        derivatives_value[in_range_index[:, 0], 0] = derivatives_erf * (1/(self.sigma * np.sqrt(2)))
        cdf[in_range_index[:, 0], 0] = erf_value
        return cdf, derivatives_value


class Skewed_Normal(Continuous_Distributions):
    def __int__(self, mu: float = None, alpha: float = None, sigma: float = None, variance: float = None, vectorized: bool = False, C: int = 1)->None:
        super(Skewed_Normal,self).__int__(mu = mu, alpha = alpha, sigma = sigma,  vectorized = vectorized, C = C)

        """
        The skewed continuous truncated gaussian distribution function
        :param alpha: the skewness parameter
        :param mu: the mean of the gaussian distribution 
        :param sigma: the standard deviation of gaussian distribution
        :param variance: the variance of gaussian distribution
        :param vectorized: the type of calculating probablity distributions
        :param C: Number of chains
        """

        if self.mu is None or self.sigma is None:
            raise Exception('The value of either mean or standard deviation is not specified (Skewed Normal distribution)!')

        self.Erf = Erf
        self.pdf = self.Prob
        self.logpdf = self.Log_prob
        self.cdf = self.CDF

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None


    def Prob(self, x)->np.ndarray:
        """
        Parallelized calculating the probablity of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probablity distribution (Cx1)
        :return: The probablity (and the derivative) of the occurance of the given variable (Cx1, Cx1)
        """
        z = (x - self.mu) / self.sigma
        erf_part, der_erf_part = 0.5 * (1 + self.Erf(z * (self.alpha / np.sqrt(2.0))))
        normal_part = (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * (z) ** 2)
        prob = 2 * erf_part * normal_part
        derivatives_prob = -np.sqrt(2/np.pi) * (z/self.sigma) * np.exp(-0.5 * (z) ** 2) * erf_part + (self.alpha/self.sigma) * np.sqrt(2/np.pi) * np.exp(-0.5 * (z) ** 2) * der_erf_part
        return prob, derivatives_prob


    def Log_prob(self, x: float)->np.ndarray:
        """
        Parallelized calculating the log (and its derivatives) of the ---- distribution
        :param x: An integer array determining the variable we are calculating its probablity distribution (Cx1)
        :return: The log probablity and derivatives of the log probablity of the occurance of an independent variable (Cx1, Cx1)
        """
        z = (x - self.mu) / self.sigma
        erf_value, der_erf_value = self.Erf((z * self.alpha)/np.sqrt(2))
        log_prob = -0.5 * np.log(2 * np.pi) - 0.5 * (z**2) + np.log(1 + erf_value)
        derivatives_log_prob = -z * (1 / self.sigma) + (1 / (self.sigma * np.sqrt(2))) * (der_erf_value / (1 + erf_value))
        return log_prob, derivatives_log_prob

    def CDF(self,x:np.ndarray)->(np.ndarray,np.ndarray):
        """
        Parallelized calculating the cumulative distribution function for ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its detivatives) with respect to the input variable (Cx1, Cx1)
        """
        return None, None




class Beta(Continuous_Distributions):
    def __int__(self, alpha: None, beta: None, vectorized: bool = False, C: int = 1)->None:
        super(Beta,self).__int__(alpha = alpha, beta = beta, vectorized = vectorized, C = C)
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
        self.pdf = self.Prob
        self.logpdf = self.Log_prob
        self.cdf = self.CDF

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None




    def Prob(self, x: np.ndarray)->(np.ndarray,np.ndarray):
        """
        Parallelized calculating the probablity of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probablity distribution (Cx1)
        :return: The probablity (and the derivative) of the occurance of the given variable (Cx1, Cx1)
        """

        x = np.clip(x, 0, 1)
        term1 = (x**(self.alpha - 1))
        term2 = ((1 - x)**(self.beta - 1))
        prob = (term1 * term2) / self.Beta(self.alpha, self.beta)
        derivatives_prob = (1/self.Beta(self.alpha, self.beta)) * (((self.alpha - 1) * x ** (self.alpha - 2)) * term2 - (self.beta - 1) * ((1 - x)**(self.beta - 2)) * term1)

        return prob, derivatives_prob

    def Log_prob(self, x: np.ndarray)->(np.ndarray,np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the ---- distribution
        :param x: An integer array determining the variable we are calculating its probablity distribution (Cx1)
        :return: The log probablity and derivatives of the log probablity of the occurance of an independent variable (Cx1, Cx1)
        """
        x = np.clip(x, 0, 1)
        log_prob = (self.alpha - 1) * np.log(x) + (self.beta - 1) * np.log(1 - x) - np.log(self.Beta(self.alpha, self.beta))
        derivatives_log_prob = ((self.alpha - 1)/x) - ((self.beta - 1)/(1 - x))
        return log_prob, derivatives_log_prob

    def CDF(self, x: np.ndarray)->(np.ndarray,np.ndarray):
        """
        Parallelized calculating the cumulative distribution function for ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its detivatives) with respect to the input variable (Cx1, Cx1)
        """
        return None, None


class Kumaraswamy(Continuous_Distributions):
    def __int__(self, alpha: None, beta: None, vectorized: bool = False, C: int = 1)->None:
        super(Kumaraswamy,self).__int__(alpha = alpha, beta = beta, vectorized = vectorized, C = C)
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

        self.pdf = self.Prob
        self.logpdf = self.Log_prob
        self.cdf = self.CDF

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def Prob(self, x: np.ndarray)-> (np.ndarray,np.ndarray):
        """
        Parallelized calculating the probablity of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probablity distribution (Cx1)
        :return: The probablity (and the derivative) of the occurance of the given variable (Cx1, Cx1)
        """
        x = np.clip(x, 0, 1)
        term1 = (x** (self.alpha-1))
        term2 = (1 - x**self.alpha)
        prob = self.beta * self.alpha * term1 * (term2 ** (self.beta - 1))
        derivatives_prob = self.beta * self.alpha * (self.alpha-1) * (x** (self.alpha - 2)) * term2 \
                           + self.beta * self.alpha * term1 * (self.beta - 1) * (-self.alpha) * (x**(self.alpha-1))* ((1 - x**self.alpha)**((self.beta - 2)))
        return prob, derivatives_prob

    def Log_prob(self, x: float)->float:
        """
        Parallelized calculating the log (and its derivatives) of the ---- distribution
        :param x: An integer array determining the variable we are calculating its probablity distribution (Cx1)
        :return: The log probablity and derivatives of the log probablity of the occurance of an independent variable (Cx1, Cx1)
        """
        x = np.clip(x, 0, 1)
        log_prob = np.log(self.alpha * self.beta) + (self.alpha - 1) * np.log(x) + (self.beta - 1) * np.log((1 - x**self.alpha))
        derivatives_log_prob = (self.alpha - 1) / x + ((self.beta - 1) * (-self.alpha * x**(self.alpha-1))) / (1 - x**self.alpha)
        return log_prob, derivatives_log_prob

    def CDF(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the cumulative distribution function for ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its detivatives) with respect to the input variable (Cx1, Cx1)
        """
        x = np.clip(x, 0, 1)
        cdf = 1 - (1 - x**self.alpha)**self.beta
        derivatives_cdf = self.beta * self.alpha * (x**(self.alpha-1)) * (1 - x**self.alpha)**(self.beta - 1)

        return cdf, derivatives_cdf




class Exponential(Continuous_Distributions):
    def __int__(self, Lambda: None, vectorized: bool = False, C: int = 1) -> None:
        super().__int__(Lambda, vectorized, C)
        """
        Initializing Kumaraswamy distribution continuous function
        :param Lambda: the rate of the change of the exponential term (Lambda>0)
        :param vectorized: boolean variable used to determine vectorized calculation
        :param C: An integer variable indicating the number of chains 
        :return: None
        """
        if self.Lambda <= 0:
            raise Exception('Parameter lambda (for calculating the beta distribution) should be positive')

        self.pdf = self.Prob
        self.logpdf = self.Log_prob
        self.cdf = self.CDF

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None


    def Prob(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the probablity of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probablity distribution (Cx1)
        :return: The probablity (and the derivative) of the occurance of the given variable (Cx1, Cx1)
        """
        prob = np.zeros((self.C, 1))
        derivatives_prob = np.zeros((self.C, 1))
        in_range_index = x>=0
        prob[in_range_index[:,0], 0] = self.Lambda * np.exp(-self.Lambda * x[in_range_index[:,0], 0])
        derivatives_prob[in_range_index[:,0], 0] = -(self.Lambda**2) * np.exp(-self.Lambda * x[in_range_index[:,0], 0])

        return prob, derivatives_prob

    def Log_prob(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the ---- distribution
        :param x: An integer array determining the variable we are calculating its probablity distribution (Cx1)
        :return: The log probablity and derivatives of the log probablity of the occurance of an independent variable (Cx1, Cx1)
        """
        in_range_index = x >= 0
        log_prob = np.ones((self.C, 1)) * -np.inf
        derivatives_log_prob = np.ones((self.C, 1)) * -np.inf
        log_prob[in_range_index[:, 0], 0] = np.log(self.Lambda) - self.Lambda * x[in_range_index[:,0], 0]
        derivatives_log_prob[in_range_index[:, 0], 0] = - self.Lambda
        return log_prob, derivatives_log_prob

    def CDF(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the cumulative distribution function for ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its detivatives) with respect to the input variable (Cx1, Cx1)
        """
        in_range_index = x >= 0
        cdf = np.zeros((self.C, 1))
        der_cdf = np.zeros((self.C, 1))
        cdf[in_range_index[:, 0], 0] = 1 - np.exp(- self.Lambda * x[in_range_index[:,0], 0])
        der_cdf[in_range_index[:, 0], 0] = self.Lambda * np.exp(-self.Lambda * x[in_range_index[:,0], 0])
        return cdf, der_cdf



class Laplace(Continuous_Distributions):
    def __int__(self, mu: None, b: None, vectorized: bool = False, C: int = 1) -> None:
        super(Laplace,self).__int__(mu = mu, b = b, vectorized = vectorized, C = C)
        """
        Initializing Kumaraswamy distribution continuous function
        :param alpha: exponent alpha parameter (alpha>0)
        :param beta:  exponent beta parameter (beta>0)
        :param vectorized: boolean variable used to determine vectorized calculation
        :param C: An integer variable indicating the number of chains 
        :return: None
        """
        if self.b <= 0:
            raise Exception('The location parameter b (for calculating the Laplace distribution) should be positive')

        self.pdf = self.Prob
        self.logpdf = self.Log_prob
        self.cdf = self.CDF

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None


    def Prob(self, x:np.ndarray,)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the probablity of the Laplace distribution
        :param x: An numpy array values determining the variable we are calculating its probablity distribution (Cx1)
        :return: The probablity (and the derivative) of the occurance of the given variable (Cx1, Cx1)
        """
        derivatives_prob = np.zeros(((self.C, 1)))
        right_index = x >= self.mu

        derivatives_prob[right_index[:, 0], 0] = (-1/(2*self.b**2)) * np.exp((-1/self.b) * (x[right_index[:, 0], 0] - self.mu))
        derivatives_prob[~right_index[:, 0], 0] = (1/(2*self.b**2)) * np.exp((1/self.b) * (x[~right_index[:, 0], 0] - self.mu))
        prob = (1/(2*self.b)) * np.exp((-1/self.b) * np.abs(x-self.mu))

        return prob, derivatives_prob

    def Log_prob(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the ---- distribution
        :param x: An integer array determining the variable we are calculating its probablity distribution (Cx1)
        :return: The log probablity and derivatives of the log probablity of the occurance of an independent variable (Cx1, Cx1)
        """

        derivatives_log_prob = np.zeros((self.C, 1))
        right_index = x >= self.mu
        derivatives_log_prob[right_index[:, 0], 0] = - (1/self.b) * (x[right_index[:, 0], 0] - self.mu)
        derivatives_log_prob[~right_index[:, 0], 0] = (1 / self.b) * (x[~right_index[:, 0], 0] - self.mu)
        log_prob = -np.log(2 * self.b) - (1/self.b) * np.abs(x - self.mu)

        return log_prob, derivatives_log_prob

    def CDF(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the cumulative distribution function for ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its detivatives) with respect to the input variable (Cx1, Cx1)
        """

        derivatives_log_prob = np.zeros(((self.C, 1)))
        right_index = x >= self.mu
        cdf = np.zeros((self.C, 1))
        cdf[right_index[:,0], 0] = 1 - 0.5 * np.exp((-1/self.b) * (x[right_index[:,0],0] - self.mu))
        cdf[~right_index[:, 0], 0] = 0.5 * np.exp((1 / self.b) * (x[~right_index[:, 0], 0] - self.mu))
        derivatives_cdf = None
        return cdf, derivatives_cdf



class Asymetric_Laplace(Continuous_Distributions):
    def __int__(self, kappa:float = None, mu:float = None, b:float = None, vectorized: bool = False, C: int = 1) -> None:
        super(myclass,self).__int__(kappa = kappa, mu = mu, b = b, vectorized = vectorized, C = C)
        """
        :param mu: The center of the distribution
        :param b : The rate of the change of the exponential term
        :param kappa: Symetric parameter
        :param vectorized: Boolean variable used to determine vectorized calculation 
        :param C: An integer variable indicating the number of chains 
        """

        if self.kappa <= 0:
            raise Exception('The values of Symmetric parameter should be positive(Asymetric Laplace distribution)!')
        if self.b<=0:
            raise Exception('The rate of the change of the exponential term should be positive(Asymetric Laplace distribution)!')

        self.pdf = self.Prob
        self.logpdf = self.Log_prob
        self.cdf = self.CDF


    @property
    def statistics(self):
        """
        Statistics calculated for the Asymetric Laplace distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def Prob(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the probablity of the Asymetric Laplace distribution
        :param x: An numpy array values determining the variable we are calculating its probablity distribution (Cx1)
        :return: The probablity (and the derivative) of the occurance of the given variable (Cx1, Cx1)
        """
        prob = np.zeros((self.C, 1))
        derivatives_prob = np.zeros((self.C, 1))
        in_range_index = x >= self.mu

        coef = self.b/(self.kappa + 1/self.kappa)
        prob[in_range_index[:,0],0] = coef * np.exp(-self.b * self.kappa * (x[in_range_index[:,0],0] - self.mu))
        prob[~in_range_index[:,0],0] = coef * np.exp((self.b / self.kappa) * (x[~in_range_index[:,0],0] - self.mu))

        derivatives_prob[in_range_index[:,0],0] = coef * (-self.b * self.kappa) * np.exp(-self.b * self.kappa * (x[in_range_index[:,0],0] - self.mu))
        derivatives_prob[~in_range_index[:, 0], 0] = coef * (self.b / self.kappa) * np.exp(-self.b * self.kappa * (x[~in_range_index[:,0],0] - self.mu))
        return prob, derivatives_prob

    def Log_prob(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the Asymetric Laplace distribution
        :param x: An integer array determining the variable we are calculating its probablity distribution (Cx1)
        :return: The log probablity and derivatives of the log probablity of the occurance of an independent variable (Cx1, Cx1)
        """

        log_prob = np.zeros((self.C, 1))
        derivatives_log_prob = np.zeros((self.C, 1))
        in_range_index = x >= self.mu
        coef = self.b / (self.kappa + 1 / self.kappa)

        log_prob[in_range_index[:,0],0] = np.log(coef) + (-self.b * self.kappa * (x[in_range_index[:,0],0] - self.mu))
        log_prob[~in_range_index[:,0],0] = np.log(coef) + ((self.b / self.kappa) * (x[~in_range_index[:,0],0] - self.mu))

        derivatives_log_prob[in_range_index[:,0],0] = -self.b * self.kappa
        derivatives_log_prob[~in_range_index[:, 0], 0] = (self.b / self.kappa)
        return log_prob, derivatives_log_prob

    def CDF(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the cumulative distribution function for Asymetric Laplace distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its detivatives) with respect to the input variable (Cx1, Cx1)
        """
        cdf = np.zeros((self.C, 1))
        in_range_index = x >= self.mu

        cdf[in_range_index[:, 0], 0] = 1 - (1/(1 + self.kappa**2)) * np.exp(-self.b * self.kappa * (x[in_range_index[:,0],0] - self.mu))
        cdf[~in_range_index[:, 0], 0] =  (self.kappa**2/(1 + self.kappa**2)) * np.exp((self.b / self.kappa) * (~x[in_range_index[:,0],0] - self.mu))
        derivatives_cdf = None
        return cdf, derivatives_cdf


class Student_t(Continuous_Distributions):
    def __int__(self, nu:float = None, mu:float = None, Lambda:float = None, vectorized: bool = False, C: int = 1) -> None:
        super(Student_t,self).__int__(nu = nu, mu = mu, Lambda = Lambda, vectorized = vectorized, C = C)
        """
        :param nu: 
        :param mu: 
        :param Lambda: 
        :param vectorized: 
        :param C: 
        :return: 
        """
        if self.nu <= 0:
            raise Exception('The value of nu should be positive (Student-t distribution)!')
        if self.sigma <= 0:
            raise Exception('The value of sigma should be positive (Student-t distribution)!')
        if self.Lambda <= 0:
            raise Exception('The value of lambda should be positive (Student-t distribution)!')


        self.Gamma = Gamma
        self.pdf = self.Prob
        self.logpdf = self.Log_prob
        self.cdf = self.CDF

    @property
    def statistics(self):
        """
        Statistics calculated for the Student_t distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def Prob(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the probablity of the Student_t distribution
        :param x: An numpy array values determining the variable we are calculating its probablity distribution (Cx1)
        :return: The probablity (and the derivative) of the occurance of the given variable (Cx1, Cx1)
        """
        coef = (self.Gamma((self.nu+1)/2) / self.Gamma(self.nu/2)) * np.sqrt(self.Lambda/(np.pi * self.nu))
        prob = coef * (1 +(self.Lambda/self.nu) * (x - self.mu)**2 )**(-(self.nu+1)/2)
        derivatives_prob = coef * (-(self.nu+1)) * (x - self.mu) * (self.Lambda/self.nu) * (1 + (self.Lambda/self.nu) * (x - self.mu)**2 )**(-(self.nu+1)/2 - 1)
        return prob, derivatives_prob

    def Log_prob(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the Student_t distribution
        :param x: An integer array determining the variable we are calculating its probablity distribution (Cx1)
        :return: The log probablity and derivatives of the log probablity of the occurance of an independent variable (Cx1, Cx1)
        """
        coef = (self.Gamma((self.nu + 1) / 2) / self.Gamma(self.nu / 2)) * np.sqrt(self.Lambda / (np.pi * self.nu))
        log_prob = np.log(coef) - ((self.nu+1)/2) * np.log(1 +(self.Lambda/self.nu) * (x - self.mu)**2)
        derivatives_log_prob = (2 * (self.Lambda/self.nu) * (x - self.mu) ) / (1 +(self.Lambda/self.nu) * (x - self.mu)**2)

        return log_prob, derivatives_log_prob

    def CDF(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the cumulative distribution function for Student_t distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its detivatives) with respect to the input variable (Cx1, Cx1)
        """
        return None, None



class Half_Student_t(Continuous_Distributions):
    def __int__(self, nu:float=None, sigma: float = None, vectorized: bool = False, C: int = 1) -> None:
        super(Half_Student_t,self).__int__(nu = nu, sigma = sigma, vectorized = vectorized, C = C)
        """
        
        :param nu: 
        :param sigma: 
        :param vectorized: 
        :param C: 
        :return: 
        """
        self.pdf = self.Prob
        self.logpdf = self.Log_prob
        self.cdf = self.CDF


    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def Prob(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the probablity of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probablity distribution (Cx1)
        :return: The probablity (and the derivative) of the occurance of the given variable (Cx1, Cx1)
        """
        coef = 2 * (self.Gamma((self.nu + 1) / 2) / self.Gamma(self.nu / 2)) * (1 / (self.sigma * np.sqrt(np.pi * self.nu)))


        prob = coef * (1 + (self.Lambda / self.nu) * (x - self.mu) ** 2) ** (-(self.nu + 1) / 2)
        derivatives_prob = coef * (-(self.nu + 1)) * (x - self.mu) * (self.Lambda / self.nu) * (1 + (self.Lambda / self.nu) * (x - self.mu) ** 2) ** (-(self.nu + 1) / 2 - 1)




        return

    def Log_prob(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the ---- distribution
        :param x: An integer array determining the variable we are calculating its probablity distribution (Cx1)
        :return: The log probablity and derivatives of the log probablity of the occurance of an independent variable (Cx1, Cx1)
        """
        return

    def CDF(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the cumulative distribution function for ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its detivatives) with respect to the input variable (Cx1, Cx1)
        """
        return














#######################################################################################################################
########################################################################################################################
#######################################################################################################################

class myclass(Continuous_Distributions):
    def __int__(self,  vectorized: bool = False, C: int = 1) -> None:
        super(myclass,self).__int__(vectorized = vectorized, C = C)
        """
        :param vectorized: 
        :param C: 
        :return: 
        """
        self.pdf = self.Prob
        self.logpdf = self.Log_prob
        self.cdf = self.CDF


    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def Prob(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the probablity of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probablity distribution (Cx1)
        :return: The probablity (and the derivative) of the occurance of the given variable (Cx1, Cx1)
        """
        return

    def Log_prob(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the ---- distribution
        :param x: An integer array determining the variable we are calculating its probablity distribution (Cx1)
        :return: The log probablity and derivatives of the log probablity of the occurance of an independent variable (Cx1, Cx1)
        """
        return

    def CDF(self, x:np.ndarray)->(np.ndarray, np.ndarray):
        """
        Parallelized calculating the cumulative distribution function for ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its detivatives) with respect to the input variable (Cx1, Cx1)
        """
        return




ts = Uniform(a=1,b=2,C=4,vectorized=True)