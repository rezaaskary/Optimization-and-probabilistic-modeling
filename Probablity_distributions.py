import numpy as np
from matplotlib.pyplot import plot, show, grid
from mathmatics import Beta, Gamma, Erf


class ContinuousDistributions:
    def __init__(self, variance: float = None, sigma: float = None, mu: float = None,
                 lb: float = None, ub: float = None, alpha: float = None,
                 a: float = None, b: float = None, vectorized: bool = True,
                 beta: float = None, Lambda: float = None, return_der_pdf: bool = True,
                 return_der_logpdf: bool = True, kappa: float = None, nu: float = None,
                 gamma: float = None) -> None:

        if isinstance(sigma, (float, int)) and isinstance(variance, (float, int)):
            raise Exception('Please Enter either variance or standard deviation!')

        if isinstance(sigma, (float, int)) and not isinstance(variance, (float, int)):
            if sigma > 0:
                self.sigma = sigma
                self.variance = sigma ** 2
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

        if isinstance(gamma, (float, int)):
            self.gamma = gamma
        elif gamma is None:
            self.gamma = None
        else:
            raise Exception('The value of nu is not specified correctly!')

        if isinstance(return_der_pdf, bool):
            self.return_der_pdf = return_der_pdf
        elif self.return_der_pdf is None:
            self.return_der_pdf = False
        else:
            raise Exception('It is not specified whether to return the derivatives of pdf!')

        if isinstance(return_der_logpdf, bool):
            self.return_der_logpdf = return_der_logpdf
        elif self.return_der_logpdf is None:
            self.return_der_logpdf = False
        else:
            raise Exception('It is not specified whether to return the derivatives of logpdf!')

    def visualize(self, lower_lim: float = -10, upper_lim: float = -10):
        """
        Visualizing the probability distribution
        :param lower_lim: the lower limit used in plotting the probability distribution
        :param upper_lim: the upper limit used in plotting the probability distribution
        :return: a line plot from matplotlib library
        """
        x_m = np.linspace(start=lower_lim, stop=upper_lim, num=1000)
        y_m = list()
        for i in range(len(x_m)):
            y_m.append(self.pdf(x_m[i]))
        plot(list(x_m.ravel()), y_m)
        grid(which='both')
        show()


class Uniform(ContinuousDistributions):
    def __init__(self, a: float = None, b: float = None, return_der_pdf: bool = True,
                 return_der_logpdf: bool = True) -> None:
        super(Uniform, self).__init__(a=a, b=b, return_der_pdf=return_der_pdf, return_der_logpdf=return_der_logpdf)
        """
        The continuous uniform distribution
        :param lb: the lower bound of the uniform distribution
        :param ub: the upper bound of the uniform distribution
        """

        if self.a >= self.b:
            raise Exception('The lower limit of the uniform distribution is greater than the upper limit!')

    @property
    def statistics(self):
        """
        Statistics calculated for the Uniform distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the probability of the Uniform distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """

        in_range_index = (x > self.a) & (x < self.b)
        prob = np.zeros_like(x)
        prob[in_range_index[:, 0], 0] = 1 / (self.b - self.a)
        if self.return_der_pdf:
            derivatives_prob = np.zeros_like(x)
        else:
            derivatives_prob = None
        return prob, derivatives_prob

    def log_prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the Uniform distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The derivatives of the log probability of the occurrence of an independent variable Cx1
        """
        in_range_index = (x > self.a) & (x < self.b)
        log_prob = -np.inf * np.ones_like(x)
        log_prob[in_range_index[:, 0], 0] = -np.log(self.b - self.a)

        if self.return_der_logpdf:
            derivatives_log_prob = -np.inf * np.ones_like(x)
            derivatives_log_prob[in_range_index[:, 0], 0] = 0
        else:
            derivatives_log_prob = None
        return log_prob, derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Uniform distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable Cx1
        """
        left_index = x <= self.a
        right_index = x >= self.a
        in_range_index = (x > self.a) & (x < self.b)
        cdf = np.ones_like(x)
        cdf[left_index[:, 0], 0] = 0
        cdf[right_index[:, 0], 0] = 1
        cdf[in_range_index[:, 0], 0] = (x[in_range_index[:, 0], 0] - self.a) / (self.b - self.a)
        return cdf


class Normal(ContinuousDistributions):
    def __init__(self, sigma: float = None, variance: float = None, mu: float = None, return_der_pdf: bool = True,
                 return_der_logpdf: bool = True) -> None:
        super(Normal, self).__init__(sigma=sigma, variance=variance, mu=mu, return_der_pdf=return_der_pdf,
                                     return_der_logpdf=return_der_logpdf)
        """
        The continuous gaussian distribution function
        :param mu: the center of the gaussian distribution
        :param std: the standard deviation of gaussian distribution
        """
        if self.mu is None or self.sigma is None:
            raise Exception('The value of either mean or standard deviation is not specified (Normal distribution)!')

        self.Erf = Erf

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the probability of the Normal distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable Cx1
        """
        prob = (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2))
        if self.return_der_pdf:
            derivatives_prob = (-1 / (self.sigma ** 3)) * np.sqrt(2 / np.pi) * (x - self.mu) * np.exp(
                -((x - self.mu) ** 2) / (2 * self.sigma ** 2))
        else:
            derivatives_prob = None
        return prob, derivatives_prob

    def log_prob(self, x: float) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the Normal distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        log_prob = -np.log(self.sigma * np.sqrt(2 * np.pi)) - ((x - self.mu) ** 2) / (2 * self.sigma ** 2)
        if self.return_der_logpdf:
            derivatives_log_prob = -(x - self.mu) / (self.sigma ** 2)
        else:
            derivatives_log_prob = None
        return log_prob, derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Normal distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function with respect to the input variable
        (Cx1, Cx1)
        """
        z = (x - self.mu) / (self.sigma * np.sqrt(2))
        erf_value, _ = self.Erf(z)
        return erf_value


class TruncatedNormal(ContinuousDistributions):
    def __init__(self, lb: float = None, ub: float = None, sigma: float = None, variance: float = None,
                 mu: float = None, return_der_pdf: bool = True, return_der_logpdf: bool = True) -> None:
        super(TruncatedNormal, self).__init__(lb=lb, ub=ub, mu=mu, sigma=sigma, variance=variance,
                                              return_der_pdf=return_der_pdf, return_der_logpdf=return_der_logpdf)

        """
        The continuous truncated gaussian distribution function
        :param lb: the lower bound of the uniform distribution
        :param ub: the upper bound of the uniform distribution
        :param mu: the center of the gaussian distribution
        :param sigma: the standard deviation of gaussian distribution
        :param variance: the variance of gaussian distribution
        """

        if self.lb >= self.ub:
            raise Exception('The lower limit of the truncated Normal distribution is greater than the upper limit!')
        if self.mu is None or self.sigma is None:
            raise Exception(
                'The value of either mean or standard deviation is not specified (Truncated Normal distribution)!')

        self.Erf = Erf

    @property
    def statistics(self):
        """
        Statistics calculated for the Truncated Normal distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the probability of the Truncated Normal distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        in_range_index = (x >= self.lb) & (x <= self.ub)
        prob = np.zeros_like(x)
        arg_r = (self.ub - self.mu) / self.sigma
        arg_l = (self.lb - self.mu) / self.sigma

        erf_r = 0.5 * (1 + self.Erf(arg_r / np.sqrt(2)))
        ert_l = 0.5 * (1 + self.Erf(arg_l / np.sqrt(2)))

        normal_argument = (x[in_range_index[:, 0], 0] - self.mu) / self.sigma
        normal_fcn_value = (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * normal_argument ** 2)
        prob[in_range_index[:, 0], 0] = (1 / self.sigma) * (normal_fcn_value / (erf_r - ert_l))
        if self.return_der_pdf:
            der_prob = np.zeros_like(x)
            der_prob[in_range_index[:, 0], 0] = (1 / self.sigma ** 2) * (1 / (erf_r - ert_l)) * (
                    -1 / (np.sqrt(2 * np.pi))) * normal_argument * np.exp(-0.5 * normal_argument ** 2)
        else:
            der_prob = None
        return prob, der_prob

    def log_prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the Truncated Normal distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """

        in_range_index = (x >= self.lb) & (x <= self.ub)
        log_prob = np.ones_like(x) * -np.inf
        arg_r = (self.ub - self.mu) / self.sigma
        arg_l = (self.lb - self.mu) / self.sigma
        normal_argument = (x[in_range_index[:, 0], 0] - self.mu) / self.sigma

        erf_r = 0.5 * (1 + self.Erf(arg_r / np.sqrt(2)))
        ert_l = 0.5 * (1 + self.Erf(arg_l / np.sqrt(2)))

        log_prob[in_range_index[:, 0], 0] = -np.log(self.sigma) - np.log(erf_r - ert_l) - 0.5 * np.log(
            2 * np.pi) - 0.5 * normal_argument ** 2

        if self.return_der_logpdf:
            derivatives_log_prob = np.ones_like(x) * -np.inf
            derivatives_log_prob[in_range_index[:, 0], 0] = (-1 / self.sigma ** 2) * (
                    x[in_range_index[:, 0], 0] - self.mu)
        else:
            derivatives_log_prob = None
        return log_prob, derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Truncated Normal distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable
        (Cx1, Cx1)
        """

        right_index = x > self.ub
        in_range_index = (x >= self.lb) & (x <= self.ub)
        cdf = np.zeros_like(x)
        cdf[right_index[:, 0], 0] = 1.0

        b = (self.ub - self.mu) / self.sigma
        a = (self.lb - self.mu) / self.sigma
        xi = (x[in_range_index[:, 0], 0] - self.mu) / self.sigma

        erf_r = 0.5 * (1 + self.Erf(b / np.sqrt(2)))
        ert_l = 0.5 * (1 + self.Erf(a / np.sqrt(2)))
        ert_xi = 0.5 * (1 + self.Erf(xi / np.sqrt(2)))
        cdf[in_range_index[:, 0], 0] = (ert_xi - ert_l) / (erf_r - ert_l)
        return cdf


class HalfNormal(ContinuousDistributions):
    def __init__(self, sigma: float = None, variance: float = None, return_der_pdf: bool = True,
                 return_der_logpdf: bool = True) -> None:
        super(HalfNormal, self).__init__(sigma=sigma, variance=variance, return_der_pdf=return_der_pdf,
                                         return_der_logpdf=return_der_logpdf)
        """
        Half Normal distribution function
        :param sigma: the standard deviation of gaussian distribution
        :param variance: the variance of gaussian distribution
        :param vectorized: the type of calculating probability distributions
        :param C: Number of chains
        """

        self.Erf = Erf

    @property
    def statistics(self):
        """
        Statistics calculated for the Half Normal distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the probability of the Half Normal distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """
        in_range_index = (x >= 0)
        prob = np.zeros_like(x)
        prob[in_range_index[:, 0], 0] = (np.sqrt(2 / np.pi) / self.sigma) * np.exp(
            -((x[in_range_index[:, 0], 0]) ** 2) / (2 * self.sigma ** 2))
        if self.return_der_pdf:
            derivatives_prob = np.zeros_like(x)
            derivatives_prob[in_range_index[:, 0], 0] = (- np.sqrt(2 / np.pi) / (self.sigma ** 3)) * (
                x[in_range_index[:, 0], 0]) * np.exp(-((x[in_range_index[:, 0], 0]) ** 2) / (2 * self.sigma ** 2))
        else:
            derivatives_prob = None
        return prob, derivatives_prob

    def log_prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the Half Normal distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """

        in_range_index = (x >= 0)
        log_prob = np.ones_like(x) * -np.inf
        log_prob[in_range_index[:, 0], 0] = 0.5 * np.log(2 / np.pi) - np.log(self.sigma) - (
                (x[in_range_index[:, 0], 0]) ** 2) / (2 * self.sigma ** 2)
        if self.return_der_logpdf:
            derivatives_log_prob = np.ones_like(x) * -np.inf
            derivatives_log_prob[in_range_index[:, 0], 0] = -x[in_range_index[:, 0], 0] / self.sigma ** 2
        else:
            derivatives_log_prob = None
        return log_prob, derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable Cx1
        """
        in_range_index = (x >= 0)
        cdf = np.zeros_like(x)
        erf_value, _ = self.Erf(x[in_range_index[:, 0], 0] / (self.sigma * np.sqrt(2)))
        cdf[in_range_index[:, 0], 0] = erf_value
        return cdf


class SkewedNormal(ContinuousDistributions):
    def __int__(self, mu: float = None, alpha: float = None, sigma: float = None, variance: float = None,
                return_der_pdf: bool = True, return_der_logpdf: bool = True) -> None:
        super(SkewedNormal, self).__init__(mu=mu, alpha=alpha, sigma=sigma, return_der_pdf=return_der_pdf,
                                           return_der_logpdf=return_der_logpdf)

        """
        The skewed continuous truncated gaussian distribution function
        :param alpha: the skewness parameter
        :param mu: the mean of the gaussian distribution 
        :param sigma: the standard deviation of gaussian distribution
        :param variance: the variance of gaussian distribution
        :param vectorized: the type of calculating probability distributions
        :param C: Number of chains
        """

        if self.mu is None or self.sigma is None:
            raise Exception(
                'The value of either mean or standard deviation is not specified (Skewed Normal distribution)!')

        self.Erf = Erf

    @property
    def statistics(self):
        """
        Statistics calculated for the Skewed Normal distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def prob(self, x) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the probability of the Skewed Normal distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """
        z = (x - self.mu) / self.sigma
        erf_part, der_erf_part = 0.5 * (1 + self.Erf(z * (self.alpha / np.sqrt(2.0))))
        normal_part = (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * (z ** 2))
        prob = 2 * erf_part * normal_part
        if self.return_der_pdf:
            derivatives_prob = -np.sqrt(2 / np.pi) * (z / self.sigma) * np.exp(-0.5 * (z ** 2)) * erf_part + (
                    self.alpha / self.sigma) * np.sqrt(2 / np.pi) * np.exp(-0.5 * (z ** 2)) * der_erf_part
        else:
            derivatives_prob = None
        return prob, derivatives_prob

    def log_prob(self, x: float) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the Skewed Normal distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        z = (x - self.mu) / self.sigma
        erf_value, der_erf_value = self.Erf((z * self.alpha) / np.sqrt(2))
        log_prob = -0.5 * np.log(2 * np.pi) - 0.5 * (z ** 2) + np.log(1 + erf_value)
        if self.return_der_logpdf:
            derivatives_log_prob = -z * (1 / self.sigma) + (1 / (self.sigma * np.sqrt(2))) * (
                    der_erf_value / (1 + erf_value))
        else:
            derivatives_log_prob = None
        return log_prob, derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Skewed Normal distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function with respect to the input variable Cx1
        """
        return None


class BetaPdf(ContinuousDistributions):
    def __init__(self, alpha: None, beta: None, return_der_pdf: bool = True, return_der_logpdf: bool = True) -> None:
        super(BetaPdf, self).__init__(alpha=alpha, beta=beta, return_der_pdf=return_der_pdf,
                                      return_der_logpdf=return_der_logpdf)
        """
        Initializing beta distribution continuous function
        :param alpha: exponent alpha parameter (alpha>0)
        :param beta:  exponent beta parameter (beta>0)
        :return: None
        """

        if self.alpha <= 0:
            raise Exception('Parameter alpha (for calculating the beta distribution) should be positive')
        if self.beta <= 0:
            raise Exception('Parameter beta (for calculating the beta distribution) should be positive')

        self.Beta = Beta

    @property
    def statistics(self):
        """
        Statistics calculated for the Beta distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the probability of the Beta distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """

        x = np.clip(a=x, a_min=0, a_max=1)
        term1 = (x ** (self.alpha - 1))
        term2 = ((1 - x) ** (self.beta - 1))
        prob = (term1 * term2) / self.Beta(self.alpha, self.beta)
        if self.return_der_pdf:
            derivatives_prob = (1 / self.Beta(self.alpha, self.beta)) * (
                    ((self.alpha - 1) * x ** (self.alpha - 2)) * term2 - (self.beta - 1) * ((1 - x) ** (self.beta - 2))
                    * term1)
        else:
            derivatives_prob = None
        return prob, derivatives_prob

    def log_prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the Beta distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        x = np.clip(a=x, a_min=0, a_max=1)
        log_prob = (self.alpha - 1) * np.log(x) + (self.beta - 1) * np.log(1 - x) - np.log(self.Beta(self.alpha,
                                                                                                     self.beta))
        if self.return_der_logpdf:
            derivatives_log_prob = ((self.alpha - 1) / x) - ((self.beta - 1) / (1 - x))
        else:
            derivatives_log_prob = None
        return log_prob, derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Beta distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function with respect to the input variable Cx1
        """
        return None


class Kumaraswamy(ContinuousDistributions):
    def __int__(self, alpha: None, beta: None, return_der_pdf: bool = True, return_der_logpdf: bool = True) -> None:
        super(Kumaraswamy, self).__init__(alpha=alpha, beta=beta, return_der_pdf=return_der_pdf,
                                          return_der_logpdf=return_der_logpdf)
        """
        Initializing Kumaraswamy distribution continuous function
        :param alpha: exponent alpha parameter (alpha>0)
        :param beta:  exponent beta parameter (beta>0)
        :return: None
        """

        if self.alpha <= 0:
            raise Exception('Parameter alpha (for calculating the beta distribution) should be positive')
        if self.beta <= 0:
            raise Exception('Parameter beta (for calculating the beta distribution) should be positive')

    @property
    def statistics(self):
        """
        Statistics calculated for the Kumaraswamy distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the probability of the Kumaraswamy distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """
        x = np.clip(a=x, a_min=0, a_max=1)
        term1 = (x ** (self.alpha - 1))
        term2 = (1 - x ** self.alpha)
        prob = self.beta * self.alpha * term1 * (term2 ** (self.beta - 1))
        if self.return_der_pdf:
            derivatives_prob = self.beta * self.alpha * (self.alpha - 1) * (x ** (self.alpha - 2)) * term2 + \
                               self.beta * self.alpha * term1 * (self.beta - 1) * (-self.alpha) * (
                                       x ** (self.alpha - 1)) * \
                               ((1 - x ** self.alpha) ** (self.beta - 2))
        else:
            derivatives_prob = None
        return prob, derivatives_prob

    def log_prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the Kumaraswamy distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        x = np.clip(a=x, a_min=0, a_max=1)
        log_prob = np.log(self.alpha * self.beta) + (self.alpha - 1) * np.log(x) + (self.beta - 1) * np.log(
            (1 - x ** self.alpha))
        if self.return_der_logpdf:
            derivatives_log_prob = (self.alpha - 1) / x + ((self.beta - 1) * (-self.alpha * x ** (self.alpha - 1))) / (
                    1 - x ** self.alpha)
        else:
            derivatives_log_prob = None
        return log_prob, derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Kumaraswamy distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable Cx1
        """
        x = np.clip(x, 0, 1)
        cdf = 1 - (1 - x ** self.alpha) ** self.beta
        return cdf


class Exponential(ContinuousDistributions):
    def __init__(self, Lambda: None, return_der_pdf: bool = True, return_der_logpdf: bool = True) -> None:
        super(Exponential, self).__init__(Lambda=Lambda, return_der_pdf=return_der_pdf,
                                          return_der_logpdf=return_der_logpdf)
        """
        Initializing Exponential distribution continuous function
        :param Lambda: the rate of the change of the exponential term (Lambda>0)
        :param vectorized: boolean variable used to determine vectorized calculation
        :param C: An integer variable indicating the number of chains 
        :return: None
        """
        if self.Lambda <= 0:
            raise Exception('Parameter lambda (for calculating the Exponential distribution) should be positive')

    @property
    def statistics(self):
        """
        Statistics calculated for the Exponential distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the probability of the Exponential distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """
        prob = np.zeros_like(x)
        in_range_index = x >= 0
        prob[in_range_index[:, 0], 0] = self.Lambda * np.exp(-self.Lambda * x[in_range_index[:, 0], 0])
        if self.return_der_pdf:
            derivatives_prob = np.zeros(x)
            derivatives_prob[in_range_index[:, 0], 0] = -(self.Lambda ** 2) * np.exp(
                -self.Lambda * x[in_range_index[:, 0], 0])
        else:
            derivatives_prob = None
        return prob, derivatives_prob

    def log_prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the Exponential distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        in_range_index = x >= 0
        log_prob = np.ones_like(x) * -np.inf
        log_prob[in_range_index[:, 0], 0] = np.log(self.Lambda) - self.Lambda * x[in_range_index[:, 0], 0]
        if self.return_der_logpdf:
            derivatives_log_prob = np.ones_like(x) * -np.inf
            derivatives_log_prob[in_range_index[:, 0], 0] = - self.Lambda
        else:
            derivatives_log_prob = None
        return log_prob, derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Exponential distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable Cx1
        """
        in_range_index = x >= 0
        cdf = np.zeros_like(x)
        cdf[in_range_index[:, 0], 0] = 1 - np.exp(- self.Lambda * x[in_range_index[:, 0], 0])
        return cdf


class Laplace(ContinuousDistributions):
    def __init__(self, mu: None, b: None, return_der_pdf: bool = True, return_der_logpdf: bool = True) -> None:
        super(Laplace, self).__init__(mu=mu, b=b, return_der_pdf=return_der_pdf, return_der_logpdf=return_der_logpdf)
        """
        Initializing Laplace distribution continuous function
        :param alpha: exponent alpha parameter (alpha>0)
        :param beta:  exponent beta parameter (beta>0)
        :param vectorized: boolean variable used to determine vectorized calculation
        :param C: An integer variable indicating the number of chains 
        :return: None
        """
        if self.b <= 0:
            raise Exception('The location parameter b (for calculating the Laplace distribution) should be positive')

    @property
    def statistics(self):
        """
        Statistics calculated for the Laplace distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def prob(self, x: np.ndarray, ) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the probability of the Laplace distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """

        right_index = x >= self.mu
        prob = (1 / (2 * self.b)) * np.exp((-1 / self.b) * np.abs(x - self.mu))
        if self.return_der_pdf:
            derivatives_prob = np.zeros_like(x)
            derivatives_prob[right_index[:, 0], 0] = (-1 / (2 * self.b ** 2)) * np.exp(
                (-1 / self.b) * (x[right_index[:, 0], 0] - self.mu))
            derivatives_prob[~right_index[:, 0], 0] = (1 / (2 * self.b ** 2)) * np.exp(
                (1 / self.b) * (x[~right_index[:, 0], 0] - self.mu))
        else:
            derivatives_prob = None
        return prob, derivatives_prob

    def log_prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the Laplace distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        right_index = x >= self.mu
        log_prob = -np.log(2 * self.b) - (1 / self.b) * np.abs(x - self.mu)
        if self.return_der_logpdf:
            derivatives_log_prob = np.zeros_like(x)
            derivatives_log_prob[right_index[:, 0], 0] = - (1 / self.b) * (x[right_index[:, 0], 0] - self.mu)
            derivatives_log_prob[~right_index[:, 0], 0] = (1 / self.b) * (x[~right_index[:, 0], 0] - self.mu)
        else:
            derivatives_log_prob = None
        return log_prob, derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Laplace distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable
        (Cx1, Cx1)
        """

        right_index = x >= self.mu
        cdf = np.zeros_like(x)
        cdf[right_index[:, 0], 0] = 1 - 0.5 * np.exp((-1 / self.b) * (x[right_index[:, 0], 0] - self.mu))
        cdf[~right_index[:, 0], 0] = 0.5 * np.exp((1 / self.b) * (x[~right_index[:, 0], 0] - self.mu))
        return cdf


class AsymmetricLaplace(ContinuousDistributions):
    def __init__(self, kappa: float = None, mu: float = None, b: float = None, return_der_pdf: bool = True,
                 return_der_logpdf: bool = True) -> None:
        super(AsymmetricLaplace, self).__init__(kappa=kappa, mu=mu, b=b, return_der_pdf=return_der_pdf,
                                                return_der_logpdf=return_der_logpdf)
        """
        :param mu: The center of the distribution
        :param b : The rate of the change of the exponential term
        :param kappa: Symmetric parameter
        """

        if self.kappa <= 0:
            raise Exception('The values of Symmetric parameter should be positive(Asymmetric Laplace distribution)!')
        if self.b <= 0:
            raise Exception(
                'The rate of the change of the exponential term should be positive(Asymmetric Laplace distribution)!')

    @property
    def statistics(self):
        """
        Statistics calculated for the Asymmetric Laplace distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the probability of the Asymmetric Laplace distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """
        prob = np.zeros_like(x)
        in_range_index = x >= self.mu

        coefficient = self.b / (self.kappa + 1 / self.kappa)
        prob[in_range_index[:, 0], 0] = coefficient * np.exp(-self.b * self.kappa * (x[in_range_index[:, 0], 0] -
                                                                                     self.mu))
        prob[~in_range_index[:, 0], 0] = coefficient * np.exp((self.b / self.kappa) * (x[~in_range_index[:, 0], 0] -
                                                                                       self.mu))
        if self.return_der_pdf:
            derivatives_prob = np.zeros_like(x)
            derivatives_prob[in_range_index[:, 0], 0] = coefficient * (-self.b * self.kappa) * np.exp(
                -self.b * self.kappa * (x[in_range_index[:, 0], 0] - self.mu))
            derivatives_prob[~in_range_index[:, 0], 0] = coefficient * (self.b / self.kappa) * np.exp(
                -self.b * self.kappa * (x[~in_range_index[:, 0], 0] - self.mu))
        else:
            derivatives_prob = None
        return prob, derivatives_prob

    def log_prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the Asymmetric Laplace distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
        (Cx1, Cx1)
        """

        log_prob = np.zeros_like(x)
        in_range_index = x >= self.mu
        coef = self.b / (self.kappa + 1 / self.kappa)

        log_prob[in_range_index[:, 0], 0] = np.log(coef) + (
                -self.b * self.kappa * (x[in_range_index[:, 0], 0] - self.mu))
        log_prob[~in_range_index[:, 0], 0] = np.log(coef) + (
                (self.b / self.kappa) * (x[~in_range_index[:, 0], 0] - self.mu))

        if self.return_der_logpdf:
            derivatives_log_prob = np.zeros_like(x)
            derivatives_log_prob[in_range_index[:, 0], 0] = -self.b * self.kappa
            derivatives_log_prob[~in_range_index[:, 0], 0] = (self.b / self.kappa)
        else:
            derivatives_log_prob = None
        return log_prob, derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Asymmetric Laplace distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable
        (Cx1, Cx1)
        """
        cdf = np.zeros_like(x)
        in_range_index = x >= self.mu

        cdf[in_range_index[:, 0], 0] = 1 - (1 / (1 + self.kappa ** 2)) * np.exp(
            -self.b * self.kappa * (x[in_range_index[:, 0], 0] - self.mu))
        cdf[~in_range_index[:, 0], 0] = (self.kappa ** 2 / (1 + self.kappa ** 2)) * np.exp(
            (self.b / self.kappa) * (~x[in_range_index[:, 0], 0] - self.mu))

        return cdf


class StudentT(ContinuousDistributions):
    def __init__(self, nu: float = None, mu: float = None, Lambda: float = None, return_der_pdf: bool = True,
                 return_der_logpdf: bool = True) -> None:
        super(StudentT, self).__init__(nu=nu, mu=mu, Lambda=Lambda, return_der_pdf=return_der_pdf,
                                       return_der_logpdf=return_der_logpdf)
        """
        :param nu: 
        :param mu: 
        :param Lambda: 
        :return: 
        """
        if self.nu <= 0:
            raise Exception('The value of nu should be positive (Student-t distribution)!')
        if self.sigma <= 0:
            raise Exception('The value of sigma should be positive (Student-t distribution)!')
        if self.Lambda <= 0:
            raise Exception('The value of lambda should be positive (Student-t distribution)!')

        self.Gamma = Gamma

    @property
    def statistics(self):
        """
        Statistics calculated for the Student_t distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the probability of the Student_t distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """
        coefficient = (self.Gamma((self.nu + 1) / 2) / self.Gamma(self.nu / 2)) *\
                      np.sqrt(self.Lambda / (np.pi * self.nu))
        prob = coefficient * (1 + (self.Lambda / self.nu) * (x - self.mu) ** 2) ** (-(self.nu + 1) / 2)
        derivatives_prob = coefficient * (-(self.nu + 1)) * (x - self.mu) * (self.Lambda / self.nu) * (
                1 + (self.Lambda / self.nu) * (x - self.mu) ** 2) ** (-(self.nu + 1) / 2 - 1)
        return prob, derivatives_prob

    def log_prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the Student_t distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        coef = (self.Gamma((self.nu + 1) / 2) / self.Gamma(self.nu / 2)) * np.sqrt(self.Lambda / (np.pi * self.nu))
        log_prob = np.log(coef) - ((self.nu + 1) / 2) * np.log(1 + (self.Lambda / self.nu) * (x - self.mu) ** 2)
        if self.return_der_logpdf:
            derivatives_log_prob = (2 * (self.Lambda / self.nu) * (x - self.mu)) / (
                    1 + (self.Lambda / self.nu) * (x - self.mu) ** 2)
        else:
            derivatives_log_prob = None
        return log_prob, derivatives_log_prob

    def cdf(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the cumulative distribution function for Student_t distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable Cx1
        """
        return None


class HalfStudentT(ContinuousDistributions):
    def __init__(self, nu: float = None, sigma: float = None, return_der_pdf: bool = True,
                 return_der_logpdf: bool = True) -> None:
        super(HalfStudentT, self).__init__(nu=nu, sigma=sigma, return_der_pdf=return_der_pdf,
                                           return_der_logpdf=return_der_logpdf)
        """
        
        :param nu: 
        :param sigma: 
        :param vectorized: 
        :param C: 
        :return: 
        """
        self.Gamma = Gamma

    @property
    def statistics(self):
        """
        Statistics calculated for the HalfStudentT distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the probability of the HalfStudentT distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """

        prob = np.zeros_like(x)
        in_range_index = x >= 0
        coef = 2 * (self.Gamma((self.nu + 1) / 2) / self.Gamma(self.nu / 2)) * (
                1 / (self.sigma * np.sqrt(np.pi * self.nu)))
        prob[in_range_index[:, 0], 0] = coef * (
                1 + (1 / self.nu) * ((x[in_range_index[:, 0], 0] / self.sigma) ** 2)) ** (-(self.nu + 1) / 2)
        if self.return_der_pdf:
            derivatives_prob = np.zeros_like(x)
            derivatives_prob[in_range_index[:, 0], 0] = coef * (-(self.nu + 1) / 2) * (
                    1 / (self.nu * self.sigma ** 2)) * (
                                                                2 * x[in_range_index[:, 0], 0]) * (
                                                                (1 + (1 / (self.nu * self.sigma ** 2)) * (
                                                                        (x[in_range_index[:, 0], 0]) ** 2)) ** (
                                                                        -(self.nu + 1) / 2 - 1))
        else:
            derivatives_prob = None
        return prob, derivatives_prob

    def log_prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the HalfStudentT distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        log_prob = np.ones_like(x) * -np.inf
        in_range_index = x >= 0
        coefficient = 2 * (self.Gamma((self.nu + 1) / 2) / self.Gamma(self.nu / 2)) * (
                1 / (self.sigma * np.sqrt(np.pi * self.nu)))

        log_prob[in_range_index[:, 0], 0] = np.log(coefficient) - ((self.nu + 1) / 2) * np.log(
            (1 + (1 / self.nu) * ((x[in_range_index[:, 0], 0] / self.sigma) ** 2)))
        if self.return_der_logpdf:
            derivatives_log_prob = np.ones_like(x) * -np.inf
            derivatives_log_prob[in_range_index[:, 0], 0] = - ((self.nu + 1) / 2) * (
                    ((2 * x[in_range_index[:, 0], 0]) / (self.nu * self.sigma ** 2)) / (
                    1 + (1 / self.nu) * ((x[in_range_index[:, 0], 0] / self.sigma) ** 2)))
        else:
            derivatives_log_prob = None
        return log_prob, derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for HalfStudentT distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable
        (Cx1, Cx1)
        """
        return None


class Cauchy(ContinuousDistributions):
    def __init__(self, gamma: float = None, mu: float = None, return_der_pdf: bool = True,
                 return_der_logpdf: bool = True) -> None:
        super(Cauchy, self).__init__(gamma=gamma, mu=mu, return_der_pdf=return_der_pdf,
                                     return_der_logpdf=return_der_logpdf)
        """
        :param vectorized: 
        :param C: 
        :return: 
        """

        if self.gamma <= 0:
            raise Exception('The value of the gamma should be positive!')

    @property
    def statistics(self):
        """
        Statistics calculated for the Cauchy  distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the probability of the Cauchy  distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """
        denominator = (1 + ((x - self.mu) / self.gamma) ** 2)
        prob = (1 / (np.pi * self.gamma)) * (1 / denominator)
        if self.return_der_pdf:
            derivatives_prob = (-2 / (np.pi * self.gamma ** 3)) * ((x - self.mu) / denominator ** 2)
        else:
            derivatives_prob = None
        return prob, derivatives_prob

    def log_prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the Cauchy  distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        denominator = (1 + ((x - self.mu) / self.gamma) ** 2)
        log_prob = -np.log(np.pi * self.gamma) - np.log(denominator)
        if self.return_der_logpdf:
            derivatives_log_prob = ((-2 / self.gamma ** 2) * (x - self.mu)) / denominator
        else:
            derivatives_log_prob = None
        return log_prob, derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Cauchy  distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable
         (Cx1, Cx1)
        """
        return


#######################################################################################################################
########################################################################################################################
#######################################################################################################################

class MyClass(ContinuousDistributions):
    def __init__(self, return_der_pdf: bool = True, return_der_logpdf: bool = True) -> None:
        super(MyClass, self).__init__(return_der_pdf=return_der_pdf, return_der_logpdf=return_der_logpdf)
        """
        :param vectorized: A boolean variable used to activate vectorized calculation 
        :param C: The number of chains used for simulation
        :return: None
        """

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the probability of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """
        pdf = np.zeros_like(x)
        if self.return_der_pdf:
            derivatives_pdf = np.zeros_like(x)
        else:
            derivatives_pdf = None

        return pdf, derivatives_pdf

    def log_pdf(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the log of the ---- distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability of the log probability of the occurrence of an independent variable Cx1
        """
        log_pdf = np.ones_like(x) * -np.inf
        if self.return_der_logpdf:
            derivatives_log_pdf = np.ones_like(x) * -np.inf
        else:
            derivatives_log_pdf = None

        return log_pdf, derivatives_log_pdf

    def cdf(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the cumulative distribution function for ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable Cx1
        """
        cdf = np.zeros_like(x)
        return cdf


ts = Uniform(a=1, b=2, return_der_pdf=True, return_der_logpdf=True)
