import numpy as np
from matplotlib.pyplot import plot, show, grid, hist, figure, subplot
import matplotlib.pyplot as plt
from mathmatics import *
RNG = np.random.default_rng(20)


class ContinuousDistributions:
    def __init__(self,
                 variance: np.ndarray = None,
                 sigma: np.ndarray = None,
                 mu: np.ndarray = None,
                 lb: np.ndarray = None,
                 ub: np.ndarray = None,
                 alpha: float = None,
                 a: np.ndarray = None,
                 b: np.ndarray = None,
                 c: np.ndarray = None,
                 beta: np.ndarray = None,
                 Lambda: np.ndarray = None,
                 kappa: np.ndarray = None,
                 nu: np.ndarray = None,
                 gamma: np.ndarray = None,
                 fixed_n_chains: bool = True,
                 chains: int = 1,
                 xm: float = None) -> None:

        if isinstance(sigma, (np.ndarray, float, int)) and isinstance(variance, (np.ndarray, float, int)):
            raise Exception('Please Enter either variance or standard deviation!')

        if isinstance(sigma, (np.ndarray, float, int)) and not isinstance(variance, (np.ndarray, float, int)):
            if sigma > 0:
                self.sigma = sigma
                self.variance = sigma ** 2
            else:
                raise Exception('The standard deviation should be a positive value!')

        if not isinstance(sigma, (np.ndarray, float, int)) and isinstance(variance, (np.ndarray, float, int)):
            if variance > 0:
                self.sigma = np.sqrt(variance)
                self.variance = variance
            else:
                raise Exception('The standard deviation should be a positive value!')

        if sigma is None and variance is None:
            self.sigma = None
            self.variance = None

        if isinstance(lb, (np.ndarray, float, int)):
            self.lb = lb
        elif lb is None:
            self.lb = None
        else:
            raise Exception('The lower bound is not specified correctly!')

        if isinstance(ub, (np.ndarray, float, int)):
            self.ub = ub
        elif ub is None:
            self.ub = None
        else:
            raise Exception('The upper bound is not specified correctly!')

        if isinstance(mu, (np.ndarray, float, int)):
            self.mu = mu
        elif mu is None:
            self.mu = None
        else:
            raise Exception('The value of mu is not specified correctly!')

        if isinstance(alpha, (np.ndarray, float, int)):
            self.alpha = alpha
        elif alpha is None:
            self.alpha = None
        else:
            raise Exception('The value of alpha is not specified correctly!')

        if isinstance(beta, (np.ndarray, float, int)):
            self.beta = beta
        elif beta is None:
            self.beta = None
        else:
            raise Exception('The value of alpha is not specified correctly!')

        if isinstance(Lambda, (np.ndarray, float, int)):
            self.Lambda = Lambda
        elif Lambda is None:
            self.Lambda = None
        else:
            raise Exception('The value of lambda is not specified correctly!')

        if isinstance(a, (np.ndarray, float, int)):
            self.a = a
        elif a is None:
            self.a = None
        else:
            raise Exception('The value of a is not specified correctly!')

        if isinstance(c, (np.ndarray, float, int)):
            self.c = c
        elif c is None:
            self.c = None
        else:
            raise Exception('The value of c is not specified correctly!')

        if isinstance(b, (np.ndarray, float, int)):
            self.b = b
        elif b is None:
            self.b = None
        else:
            raise Exception('The value of b is not specified correctly!')

        if isinstance(kappa, (np.ndarray, float, int)):
            self.kappa = kappa
        elif kappa is None:
            self.kappa = None
        else:
            raise Exception('The value of kappa is not specified correctly!')

        if isinstance(nu, (np.ndarray, float, int)):
            self.nu = nu
        elif nu is None:
            self.nu = None
        else:
            raise Exception('The value of nu is not specified correctly!')

        if isinstance(gamma, (np.ndarray, float, int)):
            self.gamma = gamma
        elif gamma is None:
            self.gamma = None
        else:
            raise Exception('The value of nu is not specified correctly!')

        if isinstance(fixed_n_chains, bool):
            self.fixed_n_chains = fixed_n_chains
        else:
            raise Exception('Please specify whether the number of chains are fixed or not !')

        if isinstance(chains, int):
            if not self.fixed_n_chains:
                raise Exception('The number of chains is specified while the variant number of chains are specified!')
            else:
                self.n_chains = chains
        elif (not isinstance(chains, int)) and self.fixed_n_chains:
            raise Exception('Please enter the number of chains(or the number of parallel evaluations) correctly!')
        else:
            print(f'-------------------------------------------------------------------------------------------------\n'
                  f'Variant number of chains is activated .'
                  f'--------------------------------------------------------------------------------------------------')

        if isinstance(xm, (float, int, float)):
            self.xm = xm
        elif xm is None:
            self.xm = None
        else:
            raise Exception('The type of xm is not entered correctly!')

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
    def __init__(self, a: float = None, b: float = None) -> None:
        """
        Continuous uniform distribution
        :param a: The lower limit of uniform distribution
        :param b: The upper limit of uniform distribution
        """
        super(Uniform, self).__init__(a=a, b=b)

        if any(self.a >= self.b):
            raise Exception('The lower limit of the uniform distribution is greater than the upper limit!')

    @property
    def statistics(self):
        """
        Statistics calculated for the Uniform distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the probability of the Uniform distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """

        in_range_index = (x > self.a) & (x < self.b)
        prob = np.zeros((len(x), 1))
        prob[in_range_index[:, 0], 0] = 1 / (self.b - self.a)
        return prob

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((len(x), 1))

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log (and its derivatives) of the Uniform distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The derivatives of the log probability of the occurrence of an independent variable Cx1
        """
        in_range_index = (x > self.a) & (x < self.b)
        log_prob = -np.inf * np.ones((len(x), 1))
        log_prob[in_range_index[:, 0], 0] = -np.log(self.b - self.a)
        return log_prob

    def log_prob_diff(self, x: np.ndarray) -> np.ndarray:
        in_range_index = (x > self.a) & (x < self.b)
        derivatives_log_prob = -np.inf * np.ones((len(x), 1))
        derivatives_log_prob[in_range_index[:, 0], 0] = 0
        return derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Uniform distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable Cx1
        """
        left_index = x <= self.a
        right_index = x >= self.b
        in_range_index = (x > self.a) & (x < self.b)
        cdf = np.ones((len(x), 1))
        cdf[left_index[:, 0], 0] = 0
        cdf[right_index[:, 0], 0] = 1
        cdf[in_range_index[:, 0], 0] = (x[in_range_index[:, 0], 0] - self.a) / (self.b - self.a)
        return cdf

    def sample(self, size: int = 100):
        sample = RNG.uniform(low=self.a, high=self.b, size=size)
        return sample


class Normal(ContinuousDistributions, ErfFcn):
    def __init__(self, sigma: float = None, variance: float = None, mu: float = None) -> None:
        """
        Normal distribution function
        :param sigma: The standard deviation of the Normal distribution (sigma>0)
        :param variance: The variance of the Normal distribution (variance>0)
        :param mu: The mean of the Normal distribution
        """
        super(Normal, self).__init__(sigma=sigma, variance=variance, mu=mu)

        if self.mu is None or self.sigma is None:
            raise Exception('The value of either mean or standard deviation is not specified (Normal distribution)!')

        # self.Erf = ErfFcn(method='simpson', intervals=10000)
        self.erf = ErfFcn(method='simpson', intervals=10000)

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
        return (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2))

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        diff_pdf = (-1 / (self.sigma ** 3)) * np.sqrt(2 / np.pi) * (x - self.mu) * np.exp(-((x - self.mu) ** 2) /
                                                                                      (2 * self.sigma ** 2))
        return diff_pdf

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log (and its derivatives) of the Normal distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        return -np.log(self.sigma * np.sqrt(2 * np.pi)) - ((x - self.mu) ** 2) / (2 * self.sigma ** 2)

    def log_prob_diff(self, x: np.ndarray) -> np.ndarray:
        return -(x - self.mu) / (self.sigma ** 2)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Normal distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function with respect to the input variable
        (Cx1, Cx1)
        """
        z = (x - self.mu) / (self.sigma * np.sqrt(2))
        erf_value = self.Erf.fcn_value(z)
        return erf_value

    def sample(self, size: int = 100):
        uniform_dis = RNG.uniform(low=0.0, high=1.0, size=size)
        return uniform_dis


class TruncatedNormal(ContinuousDistributions):
    def __init__(self, lb: float = None, ub: float = None, sigma: float = None, variance: float = None,
                 mu: float = None) -> None:
        """

        :param lb:
        :param ub:
        :param sigma:
        :param variance:
        :param mu:
        """
        super(TruncatedNormal, self).__init__(lb=lb, ub=ub, mu=mu, sigma=sigma, variance=variance)

        if self.lb >= self.ub:
            raise Exception('The lower limit of the truncated Normal distribution is greater than the upper limit!')
        if self.mu is None or self.sigma is None:
            raise Exception(
                'The value of either mean or standard deviation is not specified (Truncated Normal distribution)!')

        self.Erf = ErfFcn(method='simpson', intervals=10000)

    @property
    def statistics(self):
        """
        Statistics calculated for the Truncated Normal distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the probability of the Truncated Normal distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        ind = (x >= self.lb) & (x <= self.ub)
        arg_r = (self.ub - self.mu) / self.sigma
        arg_l = (self.lb - self.mu) / self.sigma

        erf_r = 0.5 * (1 + self.Erf.fcn_value(arg_r / np.sqrt(2)))
        ert_l = 0.5 * (1 + self.Erf.fcn_value(arg_l / np.sqrt(2)))
        normal_argument = (x[ind[:, 0], 0] - self.mu) / self.sigma
        prob = np.zeros((len(x), 1))
        normal_fcn_value = (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * normal_argument ** 2)
        prob[ind[:, 0], 0] = (1 / self.sigma) * (normal_fcn_value / (erf_r - ert_l))
        return prob

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:

        ind = (x >= self.lb) & (x <= self.ub)
        arg_r = (self.ub - self.mu) / self.sigma
        arg_l = (self.lb - self.mu) / self.sigma
        erf_r = 0.5 * (1 + self.Erf.fcn_value(arg_r / np.sqrt(2)))
        ert_l = 0.5 * (1 + self.Erf.fcn_value(arg_l / np.sqrt(2)))
        normal_argument = (x[ind[:, 0], 0] - self.mu) / self.sigma
        derivatives_prob = np.zeros((len(x), 1))
        derivatives_prob[ind[:, 0], 0] = (1 / self.sigma ** 2) * (1 / (erf_r - ert_l)) * (
                -1 / (np.sqrt(2 * np.pi))) * normal_argument * np.exp(-0.5 * normal_argument ** 2)
        return derivatives_prob

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log (and its derivatives) of the Truncated Normal distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """

        ind = (x >= self.lb) & (x <= self.ub)
        arg_r = (self.ub - self.mu) / self.sigma
        arg_l = (self.lb - self.mu) / self.sigma
        normal_argument = (x[ind[:, 0], 0] - self.mu) / self.sigma

        erf_r = 0.5 * (1 + self.Erf.fcn_value(arg_r / np.sqrt(2)))
        ert_l = 0.5 * (1 + self.Erf.fcn_value(arg_l / np.sqrt(2)))
        log_prob = np.ones((len(x), 1)) * -np.inf
        log_prob[ind[:, 0], 0] = -np.log(self.sigma) - np.log(erf_r - ert_l) - 0.5 * np.log(
            2 * np.pi) - 0.5 * normal_argument ** 2
        return log_prob

    def log_prob_diff(self, x: np.ndarray) -> np.ndarray:

        ind = (x >= self.lb) & (x <= self.ub)
        derivatives_log_prob = np.ones((len(x), 1)) * -np.inf
        derivatives_log_prob[ind[:, 0], 0] = (-1 / self.sigma ** 2) * (
                x[ind[:, 0], 0] - self.mu)
        return derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Truncated Normal distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable
        (Cx1, Cx1)
        """

        ind = x > self.ub
        in_range_index = (x >= self.lb) & (x <= self.ub)
        cdf = np.zeros((len(x), 1))
        cdf[ind[:, 0], 0] = 1.0

        b = (self.ub - self.mu) / self.sigma
        a = (self.lb - self.mu) / self.sigma
        xi = (x[in_range_index[:, 0], 0] - self.mu) / self.sigma

        erf_r = 0.5 * (1 + self.Erf.fcn_value(b / np.sqrt(2)))
        ert_l = 0.5 * (1 + self.Erf.fcn_value(a / np.sqrt(2)))
        ert_xi = 0.5 * (1 + self.Erf.fcn_value(xi / np.sqrt(2)))
        cdf[in_range_index[:, 0], 0] = (ert_xi - ert_l) / (erf_r - ert_l)
        return cdf


class HalfNormal(ContinuousDistributions):
    def __init__(self, sigma: float = None, variance: float = None) -> None:
        """

        :param sigma:
        :param variance:
        """
        super(HalfNormal, self).__init__(sigma=sigma, variance=variance)

        self.Erf = ErfFcn(method='simpson', intervals=10000)

    @property
    def statistics(self):
        """
        Statistics calculated for the Half Normal distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def prob(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the probability of the Half Normal distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """
        ind = (x >= 0)
        prob = np.zeros((len(x), 1))
        prob[ind[:, 0], 0] = (np.sqrt(2 / np.pi) / self.sigma) * np.exp(
            -((x[ind[:, 0], 0]) ** 2) / (2 * self.sigma ** 2))
        return prob

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        ind = (x >= 0)
        derivatives_prob = np.zeros((len(x), 1))
        derivatives_prob[ind[:, 0], 0] = (- np.sqrt(2 / np.pi) / (self.sigma ** 3)) * (
            x[ind[:, 0], 0]) * np.exp(-((x[ind[:, 0], 0]) ** 2) / (2 * self.sigma ** 2))
        return derivatives_prob

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log (and its derivatives) of the Half Normal distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        in_range_index = (x >= 0)
        log_prob = np.ones((len(x), 1)) * -np.inf
        log_prob[in_range_index[:, 0], 0] = 0.5 * np.log(2 / np.pi) - np.log(self.sigma) - (
                (x[in_range_index[:, 0], 0]) ** 2) / (2 * self.sigma ** 2)
        return log_prob

    def log_prob_diff(self, x: np.ndarray) -> np.ndarray:
        in_range_index = (x >= 0)
        derivatives_log_prob = np.ones((len(x), 1)) * -np.inf
        derivatives_log_prob[in_range_index[:, 0], 0] = -x[in_range_index[:, 0], 0] / self.sigma ** 2
        return derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable Cx1
        """
        in_range_index = (x >= 0)
        cdf = np.zeros((len(x), 1))
        erf_value, _ = self.Erf(x[in_range_index[:, 0], 0] / (self.sigma * np.sqrt(2)))
        cdf[in_range_index[:, 0], 0] = erf_value
        return cdf


class SkewedNormal(ContinuousDistributions):
    def __int__(self, mu: float = None, alpha: float = None, sigma: float = None, variance: float = None) -> None:
        """

        :param mu:
        :param alpha:
        :param sigma:
        :param variance:
        :param return_der_pdf:
        :param return_der_logpdf:
        :param return_pdf:
        :param return_log_pdf:
        :return:
        """
        super(SkewedNormal, self).__init__(mu=mu, alpha=alpha, sigma=sigma)

        if self.mu is None or self.sigma is None:
            raise Exception(
                'The value of either mean or standard deviation is not specified (Skewed Normal distribution)!')

        self.Erf = ErfFcn(method='simpson', intervals=10000)

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
        erf_part, der_erf_part = 0.5 * (1 + self.Erf.fcn_value(z * (self.alpha / np.sqrt(2.0))))
        normal_part = (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * (z ** 2))
        prob = 2 * erf_part * normal_part
        return prob

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mu) / self.sigma
        erf_part = 0.5 * (1 + self.Erf.fcn_value(z * (self.alpha / np.sqrt(2.0))))
        der_erf_part = self.Erf.derivatives(z * (self.alpha / np.sqrt(2.0)))
        derivatives_prob = -np.sqrt(2 / np.pi) * (z / self.sigma) * np.exp(-0.5 * (z ** 2)) * erf_part + (
                self.alpha / self.sigma) * np.sqrt(2 / np.pi) * np.exp(-0.5 * (z ** 2)) * der_erf_part
        return derivatives_prob

    def log_prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the Skewed Normal distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        z = (x - self.mu) / self.sigma
        erf_value = self.Erf.fcn_value((z * self.alpha) / np.sqrt(2))
        log_prob = -0.5 * np.log(2 * np.pi) - 0.5 * (z ** 2) + np.log(1 + erf_value)
        return log_prob

    def log_prob_diff(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mu) / self.sigma
        erf_value, der_erf_value = self.Erf.fcn_value((z * self.alpha) / np.sqrt(2))
        der_erf_value = self.Erf.derivatives((z * self.alpha) / np.sqrt(2))
        derivatives_log_prob = -z * (1 / self.sigma) + (1 / (self.sigma * np.sqrt(2))) * (
                der_erf_value / (1 + erf_value))
        return derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Skewed Normal distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function with respect to the input variable Cx1
        """
        return None


class BetaPdf(ContinuousDistributions):
    def __init__(self, alpha: None, beta: None) -> None:
        """

        :param alpha:
        :param beta:
        :param return_der_pdf:
        :param return_der_logpdf:
        :param return_pdf:
        :param return_log_pdf:
        """
        super(BetaPdf, self).__init__(alpha=alpha, beta=beta)

        if self.alpha <= 0:
            raise Exception('Parameter alpha (for calculating the beta distribution) should be positive')
        if self.beta <= 0:
            raise Exception('Parameter beta (for calculating the beta distribution) should be positive')

        self.Beta = beta_fcn

    @property
    def statistics(self):
        """
        Statistics calculated for the Beta distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def prob(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the probability of the Beta distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """

        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=1)
        term1 = (x ** (self.alpha - 1))
        term2 = ((1 - x) ** (self.beta - 1))
        prob = (term1 * term2) / self.Beta(self.alpha, self.beta)
        return prob

    def prob_diff(self, x: np.ndarray) -> np.ndarray:

        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=1)
        term1 = (x ** (self.alpha - 1))
        term2 = ((1 - x) ** (self.beta - 1))
        derivatives_prob = (1 / self.Beta(self.alpha, self.beta)) * (
                ((self.alpha - 1) * x ** (self.alpha - 2)) * term2 - (self.beta - 1) * ((1 - x) ** (self.beta - 2))
                * term1)
        return derivatives_prob

    def log_prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the Beta distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=1)
        log_prob = (self.alpha - 1) * np.log(x) + (self.beta - 1) * np.log(1 - x) - np.log(self.Beta(self.alpha,
                                                                                                     self.beta))
        return log_prob

    def log_prob_diff(self, x: np.ndarray) -> np.ndarray:

        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=1)
        derivatives_log_prob = ((self.alpha - 1) / x) - ((self.beta - 1) / (1 - x))
        return derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Beta distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function with respect to the input variable Cx1
        """
        return None


class Kumaraswamy(ContinuousDistributions):
    def __int__(self, alpha: None, beta: None, return_der_pdf: bool = True, return_der_logpdf: bool = True,
                return_pdf: bool = True, return_log_pdf: bool = True) -> None:
        """

        :param alpha:
        :param beta:
        :return:
        """
        super(Kumaraswamy, self).__init__(alpha=alpha, beta=beta)

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

    def prob(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the probability of the Kumaraswamy distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=1)
        term1 = (x ** (self.alpha - 1))
        term2 = (1 - x ** self.alpha)
        prob = self.beta * self.alpha * term1 * (term2 ** (self.beta - 1))
        return prob

    def prob_diff(self, x: np.ndarray) -> np.ndarray:

        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=1)
        term1 = (x ** (self.alpha - 1))
        term2 = (1 - x ** self.alpha)
        derivatives_prob = self.beta * self.alpha * (self.alpha - 1) * (x ** (self.alpha - 2)) * term2 + \
                           self.beta * self.alpha * term1 * (self.beta - 1) * (-self.alpha) * (
                                   x ** (self.alpha - 1)) * \
                           ((1 - x ** self.alpha) ** (self.beta - 2))
        return derivatives_prob

    def log_prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the Kumaraswamy distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=1)
        log_prob = np.log(self.alpha * self.beta) + (self.alpha - 1) * np.log(x) + (self.beta - 1) * np.log(
            (1 - x ** self.alpha))

        return log_prob

    def log_prob_diff(self, x: np.ndarray) -> np.ndarray:
        """

        :param x:
        :return:
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=1)
        derivatives_log_prob = (self.alpha - 1) / x + ((self.beta - 1) * (-self.alpha * x ** (self.alpha - 1))) / (
                1 - x ** self.alpha)

        return derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Kumaraswamy distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable Cx1
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=1)
        cdf = 1 - (1 - x ** self.alpha) ** self.beta
        return cdf


class Exponential(ContinuousDistributions):
    def __init__(self, Lambda: None) -> None:
        """

        :param Lambda:
        :param return_der_pdf:
        :param return_der_logpdf:
        :param return_pdf:
        :param return_log_pdf:
        """
        super(Exponential, self).__init__(Lambda=Lambda)

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
        in_range_index = x >= 0
        prob = np.zeros((len(x), 1))
        prob[in_range_index[:, 0], 0] = self.Lambda * np.exp(-self.Lambda * x[in_range_index[:, 0], 0])
        return prob

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        in_range_index = x >= 0
        derivatives_prob = np.zeros((len(x), 1))
        derivatives_prob[in_range_index[:, 0], 0] = -(self.Lambda ** 2) * np.exp(
            -self.Lambda * x[in_range_index[:, 0], 0])
        return derivatives_prob

    def log_prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the Exponential distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        in_range_index = x >= 0
        log_prob = np.ones((len(x), 1)) * -np.inf
        log_prob[in_range_index[:, 0], 0] = np.log(self.Lambda) - self.Lambda * x[in_range_index[:, 0], 0]

        return log_prob

    def log_prob_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log (and its derivatives) of the Exponential distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        in_range_index = x >= 0
        derivatives_log_prob = np.ones((len(x), 1)) * -np.inf
        derivatives_log_prob[in_range_index[:, 0], 0] = - self.Lambda
        return derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Exponential distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable Cx1
        """
        in_range_index = x >= 0
        cdf = np.zeros((len(x), 1))
        cdf[in_range_index[:, 0], 0] = 1 - np.exp(- self.Lambda * x[in_range_index[:, 0], 0])
        return cdf


class Laplace(ContinuousDistributions):
    def __init__(self, mu: None, b: None) -> None:
        """

        :param mu:
        :param b:
        """
        super(Laplace, self).__init__(mu=mu, b=b)

        if self.b <= 0:
            raise Exception('The location parameter b (for calculating the Laplace distribution) should be positive')

    @property
    def statistics(self):
        """
        Statistics calculated for the Laplace distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def prob(self, x: np.ndarray, ) -> np.ndarray:
        """
        Parallelized calculating the probability of the Laplace distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """

        prob = (1 / (2 * self.b)) * np.exp((-1 / self.b) * np.abs(x - self.mu))
        return prob

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        right_index = x >= self.mu
        derivatives_prob = np.zeros((len(x), 1))
        derivatives_prob[right_index[:, 0], 0] = (-1 / (2 * self.b ** 2)) * np.exp(
            (-1 / self.b) * (x[right_index[:, 0], 0] - self.mu))
        derivatives_prob[~right_index[:, 0], 0] = (1 / (2 * self.b ** 2)) * np.exp(
            (1 / self.b) * (x[~right_index[:, 0], 0] - self.mu))
        return derivatives_prob

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log (and its derivatives) of the Laplace distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        right_index = x >= self.mu
        log_prob = -np.log(2 * self.b) - (1 / self.b) * np.abs(x - self.mu)
        return log_prob

    def log_prob_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log (and its derivatives) of the Laplace distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        right_index = x >= self.mu
        derivatives_log_prob = np.zeros((len(x), 1))
        derivatives_log_prob[right_index[:, 0], 0] = - (1 / self.b) * (x[right_index[:, 0], 0] - self.mu)
        derivatives_log_prob[~right_index[:, 0], 0] = (1 / self.b) * (x[~right_index[:, 0], 0] - self.mu)
        return derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Laplace distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable
        (Cx1, Cx1)
        """

        right_index = x >= self.mu
        cdf = np.zeros((len(x), 1))
        cdf[right_index[:, 0], 0] = 1 - 0.5 * np.exp((-1 / self.b) * (x[right_index[:, 0], 0] - self.mu))
        cdf[~right_index[:, 0], 0] = 0.5 * np.exp((1 / self.b) * (x[~right_index[:, 0], 0] - self.mu))
        return cdf


class AsymmetricLaplace(ContinuousDistributions):
    def __init__(self, kappa: float = None, mu: float = None, b: float = None) -> None:
        """

        :param kappa:
        :param mu:
        :param b:
        """
        super(AsymmetricLaplace, self).__init__(kappa=kappa, mu=mu, b=b)

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

    def prob(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the probability of the Asymmetric Laplace distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """
        in_range_index = x >= self.mu
        coefficient = self.b / (self.kappa + 1 / self.kappa)
        prob = np.zeros((len(x), 1))
        prob[in_range_index[:, 0], 0] = coefficient * np.exp(-self.b * self.kappa * (x[in_range_index[:, 0], 0] -
                                                                                     self.mu))
        prob[~in_range_index[:, 0], 0] = coefficient * np.exp((self.b / self.kappa) * (x[~in_range_index[:, 0], 0] -
                                                                                       self.mu))
        return prob

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:

        in_range_index = x >= self.mu
        coefficient = self.b / (self.kappa + 1 / self.kappa)
        derivatives_prob = np.zeros((len(x), 1))
        derivatives_prob[in_range_index[:, 0], 0] = coefficient * (-self.b * self.kappa) * np.exp(
            -self.b * self.kappa * (x[in_range_index[:, 0], 0] - self.mu))
        derivatives_prob[~in_range_index[:, 0], 0] = coefficient * (self.b / self.kappa) * np.exp(
            -self.b * self.kappa * (x[~in_range_index[:, 0], 0] - self.mu))
        return derivatives_prob

    def log_prob(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the log (and its derivatives) of the Asymmetric Laplace distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
        (Cx1, Cx1)
        """
        in_range_index = x >= self.mu
        log_prob = np.zeros((len(x), 1))
        coef = self.b / (self.kappa + 1 / self.kappa)
        log_prob[in_range_index[:, 0], 0] = np.log(coef) + (
                -self.b * self.kappa * (x[in_range_index[:, 0], 0] - self.mu))
        log_prob[~in_range_index[:, 0], 0] = np.log(coef) + (
                (self.b / self.kappa) * (x[~in_range_index[:, 0], 0] - self.mu))
        return log_prob

    def log_prob_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log (and its derivatives) of the Asymmetric Laplace distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
        (Cx1, Cx1)
        """
        in_range_index = x >= self.mu
        derivatives_log_prob = np.zeros((len(x), 1))
        derivatives_log_prob[in_range_index[:, 0], 0] = -self.b * self.kappa
        derivatives_log_prob[~in_range_index[:, 0], 0] = (self.b / self.kappa)
        return derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Asymmetric Laplace distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable
        (Cx1, Cx1)
        """
        cdf = np.zeros((len(x), 1))
        in_range_index = x >= self.mu
        cdf[in_range_index[:, 0], 0] = 1 - (1 / (1 + self.kappa ** 2)) * np.exp(
            -self.b * self.kappa * (x[in_range_index[:, 0], 0] - self.mu))
        cdf[~in_range_index[:, 0], 0] = (self.kappa ** 2 / (1 + self.kappa ** 2)) * np.exp(
            (self.b / self.kappa) * (~x[in_range_index[:, 0], 0] - self.mu))
        return cdf


class StudentT(ContinuousDistributions):
    def __init__(self, nu: float = None, mu: float = None, Lambda: float = None) -> None:
        """

        :param nu:
        :param mu:
        :param Lambda:
        :param return_der_pdf:
        :param return_der_logpdf:
        :param return_pdf:
        :param return_log_pdf:
        """
        super(StudentT, self).__init__(nu=nu, mu=mu, Lambda=Lambda)

        if self.nu <= 0:
            raise Exception('The value of nu should be positive (Student-t distribution)!')
        if self.sigma <= 0:
            raise Exception('The value of sigma should be positive (Student-t distribution)!')
        if self.Lambda <= 0:
            raise Exception('The value of lambda should be positive (Student-t distribution)!')

        self.Gamma = gamma_fcn

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
        coefficient = (self.Gamma((self.nu + 1) / 2) / self.Gamma(self.nu / 2)) * \
                      np.sqrt(self.Lambda / (np.pi * self.nu))
        prob = coefficient * (1 + (self.Lambda / self.nu) * (x - self.mu) ** 2) ** (-(self.nu + 1) / 2)
        return prob

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:

        coefficient = (self.Gamma((self.nu + 1) / 2) / self.Gamma(self.nu / 2)) * \
                      np.sqrt(self.Lambda / (np.pi * self.nu))
        derivatives_prob = coefficient * (-(self.nu + 1)) * (x - self.mu) * (self.Lambda / self.nu) * (
                1 + (self.Lambda / self.nu) * (x - self.mu) ** 2) ** (-(self.nu + 1) / 2 - 1)
        return derivatives_prob

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log (and its derivatives) of the Student_t distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        coef = (self.Gamma((self.nu + 1) / 2) / self.Gamma(self.nu / 2)) * np.sqrt(self.Lambda / (np.pi * self.nu))
        log_prob = np.log(coef) - ((self.nu + 1) / 2) * np.log(1 + (self.Lambda / self.nu) * (x - self.mu) ** 2)

        return log_prob

    def log_prob_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log (and its derivatives) of the Student_t distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        derivatives_log_prob = (2 * (self.Lambda / self.nu) * (x - self.mu)) / (
                1 + (self.Lambda / self.nu) * (x - self.mu) ** 2)
        return derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Student_t distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable Cx1
        """
        return None


class HalfStudentT(ContinuousDistributions):
    def __init__(self, nu: float = None, sigma: float = None) -> None:
        """

        :param nu:
        :param sigma:
        :param return_der_pdf:
        :param return_der_logpdf:
        :param return_pdf:
        :param return_log_pdf:
        """
        super(HalfStudentT, self).__init__(nu=nu, sigma=sigma)

        self.Gamma = gamma_fcn

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
        in_range_index = x >= 0
        coef = 2 * (self.Gamma((self.nu + 1) / 2) / self.Gamma(self.nu / 2)) * (
                1 / (self.sigma * np.sqrt(np.pi * self.nu)))
        prob = np.zeros((len(x), 1))
        prob[in_range_index[:, 0], 0] = coef * (
                1 + (1 / self.nu) * ((x[in_range_index[:, 0], 0] / self.sigma) ** 2)) ** (-(self.nu + 1) / 2)
        return prob

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:

        in_range_index = x >= 0
        coef = 2 * (self.Gamma((self.nu + 1) / 2) / self.Gamma(self.nu / 2)) * (
                1 / (self.sigma * np.sqrt(np.pi * self.nu)))
        derivatives_prob = np.zeros((len(x), 1))
        derivatives_prob[in_range_index[:, 0], 0] = coef * (-(self.nu + 1) / 2) * (
                1 / (self.nu * self.sigma ** 2)) * (2 * x[in_range_index[:, 0], 0]) * (
                                                            (1 + (1 / (self.nu * self.sigma ** 2)) * (
                                                                    (x[in_range_index[:, 0], 0]) ** 2)) ** (
                                                                    -(self.nu + 1) / 2 - 1))
        return derivatives_prob

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log (and its derivatives) of the HalfStudentT distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """

        in_range_index = x >= 0
        log_prob = np.ones((len(x), 1)) * -np.inf
        coefficient = 2 * (self.Gamma((self.nu + 1) / 2) / self.Gamma(self.nu / 2)) * (
                1 / (self.sigma * np.sqrt(np.pi * self.nu)))
        log_prob[in_range_index[:, 0], 0] = np.log(coefficient) - ((self.nu + 1) / 2) * np.log(
            (1 + (1 / self.nu) * ((x[in_range_index[:, 0], 0] / self.sigma) ** 2)))
        return log_prob

    def log_prob_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log (and its derivatives) of the HalfStudentT distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        in_range_index = x >= 0
        derivatives_log_prob = np.ones((len(x), 1)) * -np.inf
        derivatives_log_prob[in_range_index[:, 0], 0] = - ((self.nu + 1) / 2) * (
                ((2 * x[in_range_index[:, 0], 0]) / (self.nu * self.sigma ** 2)) / (
                1 + (1 / self.nu) * ((x[in_range_index[:, 0], 0] / self.sigma) ** 2)))
        return derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for HalfStudentT distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable
        (Cx1, Cx1)
        """
        return None


class Cauchy(ContinuousDistributions):
    def __init__(self, gamma: float = None, mu: float = None) -> None:
        """

        :param gamma:
        :param mu:
        :param return_der_pdf:
        :param return_der_logpdf:
        :param return_pdf:
        :param return_log_pdf:
        """
        super(Cauchy, self).__init__(gamma=gamma, mu=mu)

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
        return prob

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        denominator = (1 + ((x - self.mu) / self.gamma) ** 2)
        derivatives_prob = (-2 / (np.pi * self.gamma ** 3)) * ((x - self.mu) / denominator ** 2)
        return derivatives_prob

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log (and its derivatives) of the Cauchy  distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        denominator = (1 + ((x - self.mu) / self.gamma) ** 2)
        log_prob = -np.log(np.pi * self.gamma) - np.log(denominator)
        return log_prob

    def log_prob_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log (and its derivatives) of the Cauchy  distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability and derivatives of the log probability of the occurrence of an independent variable
         (Cx1, Cx1)
        """
        denominator = (1 + ((x - self.mu) / self.gamma) ** 2)
        derivatives_log_prob = ((-2 / self.gamma ** 2) * (x - self.mu)) / denominator
        return derivatives_log_prob

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for Cauchy  distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable
         (Cx1, Cx1)
        """
        return


class HalfCauchy(ContinuousDistributions):
    def __init__(self, beta: float = None) -> None:
        """
        :param beta:
        :param return_der_pdf:
        :param return_der_logpdf:
        :param return_pdf:
        :param return_log_pdf:
        """
        super(HalfCauchy, self).__init__(beta=beta)

        if self.beta <= 0:
            raise Exception('The value of beta should be positive (Half Couchy)!')
        self.atan = arctan_fcn

    @property
    def statistics(self):
        """
        Statistics calculated for the HalfCauchy distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the probability of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """

        index_in_range = x >= 0
        denominator = (1 + ((x[index_in_range[:, 0], 0]) / self.beta) ** 2)
        pdf = np.zeros((len(x), 1))
        pdf[index_in_range[:, 0], 0] = (2 / (self.beta * np.pi)) * (1 / denominator)

        return pdf

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:

        index_in_range = x >= 0
        denominator = (1 + ((x[index_in_range[:, 0], 0]) / self.beta) ** 2)
        derivatives_pdf = np.zeros((len(x), 1))
        derivatives_pdf[index_in_range[:, 0], 0] = (-4 / ((self.beta ** 3) * np.pi)) * (
                (x[index_in_range[:, 0], 0]) /
                denominator ** 2)
        return derivatives_pdf

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log of the Half Cauchy distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability of the log probability of the occurrence of an independent variable Cx1
        """
        index_in_range = x >= 0
        denominator = (1 + ((x[index_in_range[:, 0], 0]) / self.beta) ** 2)
        log_pdf = np.ones((len(x), 1)) * -np.inf
        log_pdf[index_in_range[:, 0], 0] = np.log(2 / (self.beta * np.pi)) - np.log(denominator)

        return log_pdf

    def log_pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log of the Half Cauchy distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability of the log probability of the occurrence of an independent variable Cx1
        """
        index_in_range = x >= 0
        denominator = (1 + ((x[index_in_range[:, 0], 0]) / self.beta) ** 2)
        derivatives_log_pdf = np.ones((len(x), 1)) * -np.inf
        derivatives_log_pdf[index_in_range[:, 0], 0] = (-2 / self.beta ** 2) * (x[index_in_range[:, 0], 0] /
                                                                                denominator)
        return derivatives_log_pdf

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function for ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable Cx1
        """
        cdf = np.zeros((len(x), 1))
        index_in_range = x >= 0
        cdf[index_in_range[:, 0], 0] = (2 / np.pi) * self.atan(x[index_in_range[:, 0], 0] / self.beta)
        return cdf


class GammaDistribution(ContinuousDistributions):
    def __init__(self, alpha: float = None, beta: float = None) -> None:
        """

        :param alpha:
        :param beta:
        :param return_der_pdf:
        :param return_der_logpdf:
        :param return_pdf:
        :param return_log_pdf:
        """
        super(GammaDistribution, self).__init__(alpha=alpha, beta=beta)

        self.LowerGamma = lower_incomplete_gamma_fcn
        self.Gamma = gamma_fcn

    @property
    def statistics(self):
        """
        Statistics calculated for the Gamma distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the probability of the Gamma distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        coefficient = ((self.beta ** self.alpha) / self.Gamma(self.alpha))
        pdf = coefficient * (x ** (self.alpha - 1)) * (np.exp(-self.beta * x))
        return pdf

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:

        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        coefficient = ((self.beta ** self.alpha) / self.Gamma(self.alpha))
        derivatives_pdf = coefficient * ((self.alpha - 1) * (x ** (self.alpha - 2)) * np.exp(-self.beta * x)) + \
                          coefficient * ((-self.beta) * (x ** (self.alpha - 1)) * np.exp(-self.beta * x))
        return derivatives_pdf

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log of the Gamma distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability of the log probability of the occurrence of an independent variable Cx1
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        coefficient = ((self.beta ** self.alpha) / self.Gamma(self.alpha))
        log_pdf = np.log(coefficient) + (self.alpha - 1) * np.log(x) - self.beta * x
        return log_pdf

    def log_pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log of the Gamma distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability of the log probability of the occurrence of an independent variable Cx1
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        derivatives_log_pdf = (self.alpha - 1) / x - self.beta
        return derivatives_log_pdf

    def cdf(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the cumulative distribution function for ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable Cx1
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        cdf = np.zeros((len(x), 1))
        self.LowerGamma(self.alpha, self.beta * x) / self.Gamma(self.alpha)
        return cdf


class InverseGamma(ContinuousDistributions):
    def __init__(self, alpha: float = None, beta: float = None) -> None:
        """

        :param alpha:
        :param beta:
        :param return_der_pdf:
        :param return_der_logpdf:
        :param return_pdf:
        :param return_log_pdf:
        """
        super(InverseGamma, self).__init__(alpha=alpha, beta=beta)

        if self.alpha <= 0:
            raise Exception('The value of alpha should be positive (InverseGamma)!')
        if self.beta <= 0:
            raise Exception('The value of beta should be positive (InverseGamma)!')

        self.Gamma = gamma_fcn

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the probability of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        coefficient = ((self.beta ** self.alpha) / self.Gamma(self.alpha))
        pdf = coefficient * (x ** (-self.alpha - 1)) * (np.exp(-self.beta / x))
        return pdf

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        coefficient = ((self.beta ** self.alpha) / self.Gamma(self.alpha))
        derivatives_pdf = coefficient * ((-self.alpha - 1) * (x ** (-self.alpha - 2)) * np.exp(-self.beta / x)) + \
                          coefficient * ((self.beta / x ** 2) * (x ** (-self.alpha - 1)) * np.exp(-self.beta / x))

        return derivatives_pdf

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log of the ---- distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability of the log probability of the occurrence of an independent variable Cx1
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        coefficient = ((self.beta ** self.alpha) / self.Gamma(self.alpha))
        log_pdf = np.log(coefficient) + (-self.alpha - 1) * np.log(x) - self.beta / x
        return log_pdf

    def log_pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log of the ---- distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability of the log probability of the occurrence of an independent variable Cx1
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        derivatives_log_pdf = (-self.alpha - 1) / x + self.beta / x ** 2
        return derivatives_log_pdf

    def cdf(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Parallelized calculating the cumulative distribution function for ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function (and its derivatives) with respect to the input variable Cx1
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        cdf = None
        return cdf



class Weibull(ContinuousDistributions):
    def __init__(self, kappa: float = None, Lambda: float = None) -> None:
        super(Weibull, self).__init__(kappa=kappa, Lambda=Lambda)

        if self.kappa <= 0:
            raise  Exception('The value of kappa should be positive (Weibull distribution)!')

        if self.Lambda <= 0:
            raise  Exception('The value of lambda should be positive (Weibull distribution)!')

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the probability of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1)
        """
        # x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        pdf = np.zeros((len(x), 1))
        index_in_range = x >= 0
        pdf[index_in_range[:, 0], 0] = (self.kappa/self.Lambda) *\
                                       ((x[index_in_range[:, 0], 0]/self.Lambda)**(self.kappa-1)) *\
                                       np.exp(-(x[index_in_range[:, 0], 0]/self.Lambda)**self.kappa)
        return pdf

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the  ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The derivative of the probability distribution (Cx1)
        """
        derivatives_pdf = np.zeros((len(x), 1))
        index_in_range = x >= 0
        exp_term = np.exp(-(x[index_in_range[:, 0], 0]/self.Lambda)**self.kappa)
        polynomial_term = ((x[index_in_range[:, 0], 0]/self.Lambda) ** (self.kappa-1))
        derivatives_pdf[index_in_range[:, 0], 0] = (self.kappa / self.Lambda) * ((((self.kappa-1) / self.Lambda) * (x[index_in_range[:, 0], 0]/self.Lambda) ** (self.kappa-2)) * exp_term)\
                          - ((self.kappa / self.Lambda)**2) * (((x[index_in_range[:, 0], 0]/self.Lambda) ** (2*self.kappa-2))) * exp_term

        return derivatives_pdf

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log probablity of ---- distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability of the log probability of ---- distribution (Cx1)
        """
        log_pdf = np.ones((len(x), 1)) * -np.inf
        index_in_range = x >= 0

        log_pdf[index_in_range[:, 0], 0] = np.log(self.kappa/self.Lambda) +\
                                           (self.kappa-1)*np.log(x[index_in_range[:, 0], 0]/self.Lambda) -\
                                           (x[index_in_range[:, 0], 0]/self.Lambda)**self.kappa


        return log_pdf

    def log_pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the log of the ---- distribution
        :param x: An input array of the probability distribution function(Cx1)
        :return: The log probability of the log probability of the occurrence of an independent variable (Cx1)
        """
        derivatives_log_pdf = np.ones((len(x), 1)) * -np.inf
        index_in_range = x >= 0

        derivatives_log_pdf[index_in_range[:, 0], 0] = (self.kappa-1)/x[index_in_range[:, 0], 0] -\
                                                       (self.kappa * (1/self.Lambda)**self.kappa) *\
                                                       (x[index_in_range[:, 0], 0])**(self.kappa-1)
        return derivatives_log_pdf

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function of ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function of ---- distribution (Cx1)
        """
        cdf = np.zeros((len(x), 1))
        index_in_range = x >= 0
        cdf[index_in_range[:, 0], 0] = 1-np.exp(-(x[index_in_range[:, 0], 0]/self.Lambda)**self.kappa)
        return cdf


class ChiSquared(ContinuousDistributions):
    def __init__(self, kappa: int = None) -> None:
        super(ChiSquared, self).__init__(kappa=kappa)

        if self.kappa <= 0:
            raise Exception('The degree of freedom should be positive integer (Chi Squared distribution)!')
        if not isinstance(self.kappa, int):
            raise Exception('The degree of freedom should be positive integer (Chi Squared distribution)!')

        self.Gamma = gamma_fcn
        self.ligf = lower_incomplete_gamma_fcn

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the probability of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        pdf = (1/2**(self.kappa/2)) * (1/self.Gamma(self.kappa/2)) * (x**(0.5*(self.kappa-2))) * np.exp(-x/2)
        return pdf

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the  ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The derivative of the probability distribution (Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        derivatives_pdf = (1/2**(self.kappa/2)) * (1/self.Gamma(self.kappa/2)) * np.exp(-x/2) *\
                      ((self.kappa/2-1) * (x**(0.5*self.kappa-2)) - 0.5* (x**(0.5*self.kappa-1)))
        return derivatives_pdf

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log probablity of ---- distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability of the log probability of ---- distribution (Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        log_pdf = -(self.kappa/2)*np.log(2) - np.log(self.Gamma(self.kappa / 2)) + (self.kappa / 2 -1) * np.log(x) - x/2
        return log_pdf

    def log_pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the log of the ---- distribution
        :param x: An input array of the probability distribution function(Cx1)
        :return: The log probability of the log probability of the occurrence of an independent variable (Cx1)
        """

        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        derivatives_log_pdf = (self.kappa / 2 - 1)/x - 1/2

        return derivatives_log_pdf

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function of ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function of ---- distribution (Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        cdf = self.ligf(self.kappa/2, 0.5*x)/self.Gamma(self.kappa/2)
        return cdf

class LogNormal(ContinuousDistributions):
    def __init__(self, mu: float = None, sigma: float = None, variance: float = None) -> None:
        super(LogNormal, self).__init__(mu=mu, sigma=sigma, variance=variance)


        if self.sigma <= 0 :
            raise Exception('The value of the standard deviation should be positive (Log Normal distribution)!')

        if self.variance <= 0 :
            raise Exception('The value of variance should be positive (Log Normal distribution)!')

        self.erf = erf_fcn

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the probability of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        pdf = (1/(x*self.sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*(1/self.sigma**2)*(np.log(x)-self.mu)**2)
        return pdf

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the  ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The derivative of the probability distribution (Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        derivatives_pdf = (-1/x**2) * (1/(self.sigma*np.sqrt(2*np.pi))) *\
                      np.exp(-0.5*(1/self.sigma**2)*(np.log(x)-self.mu)**2) *\
                      (1 +(1/(self.sigma**2))*(np.log(x)-self.mu))
        return derivatives_pdf

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log probablity of ---- distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability of the log probability of ---- distribution (Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        log_pdf = -np.log((x * self.sigma * np.sqrt(2 * np.pi))) - 0.5 * ((1 / self.sigma ** 2) * ((np.log(x) - self.mu) ** 2))
        return log_pdf

    def log_pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the log of the ---- distribution
        :param x: An input array of the probability distribution function(Cx1)
        :return: The log probability of the log probability of the occurrence of an independent variable (Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        derivatives_log_pdf = (-1/x)*(1+(1 / (self.sigma ** 2)) * (np.log(x) - self.mu))
        return derivatives_log_pdf

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function of ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function of ---- distribution (Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        input_argument = (np.log(x)-self.mu)/(self.sigma*np.sqrt(2))
        cdf = 0.5*(1+self.erf(input_argument))
        return cdf


class Wald (ContinuousDistributions):
    def __init__(self, mu: float = None, Lambda: float = None) -> None:
        """

        :param mu:
        :param Lambda:
        """
        super(Wald, self).__init__(mu=mu, Lambda=Lambda)
        if self.Lambda <= 0 :
            raise Exception('The value of lambda should be positive (Wald distribution)!')

        if self.mu <= 0 :
            raise Exception('The value of mean should be positive (Log Normal distribution)!')

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the probability of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        pdf = np.sqrt(self.Lambda/(2*np.pi* x ** 3)) * np.exp((-self.Lambda/(2*x*self.mu**2))*(x-self.mu)**2)
        return pdf

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the  ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The derivative of the probability distribution (Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        exp_term =  np.exp((-self.Lambda/(2*x*self.mu**2))*(x-self.mu)**2)
        coefficient =  np.sqrt(self.Lambda/(2*np))
        derivatives_pdf = -1.5 * coefficient * (x ** -2.5) * exp_term + coefficient * (x ** -1.5) * exp_term *\
                          (-self.Lambda/(2*self.mu**2)) * (2*(1-self.mu/x)-(1-self.mu/x)**2)
        return derivatives_pdf

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log probablity of ---- distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability of the log probability of ---- distribution (Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        log_pdf =  0.5 * np.log(self.Lambda / (2 * np.pi * x ** 3)) + ((-self.Lambda / (2 * x * self.mu ** 2)) * (x - self.mu) ** 2)
        return log_pdf

    def log_pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the log of the ---- distribution
        :param x: An input array of the probability distribution function(Cx1)
        :return: The log probability of the log probability of the occurrence of an independent variable (Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        derivatives_log_pdf = (-1.5/x) + (-self.Lambda/(2*self.mu**2)) * (2*(1-self.mu/x)-(1-self.mu/x)**2)
        return derivatives_log_pdf

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function of ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function of ---- distribution (Cx1)
        """
        def normal_fcn(z):
            return (1/np.sqrt(2*np.pi))*np.exp(-0.5*z**2)

        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=np.inf)
        cdf = normal_fcn(np.sqrt(self.Lambda/x)*(x/self.mu-1)) + np.exp((2*self.Lambda)/self.mu)*\
              normal_fcn(-np.sqrt(self.Lambda/x)*(x/self.mu+1))
        return cdf



class Pareto(ContinuousDistributions):
    def __init__(self, xm: float = None, alpha: float = None) -> None:
        """

        :param xm:
        :param alpha:
        """
        super(MyClass, self).__init__(xm=xm, alpha=alpha)

        if self.xm <= 0 :
            raise Exception('The value of xm should be positive (Pareto distribution)!')

        if self.alpha <= 0 :
            raise Exception('The value of alpha should be positive (Pareto distribution)!')

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the probability of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1)
        """

        pdf = np.zeros((len(x), 1))
        in_range_index = x >= self.xm
        pdf[in_range_index[:, 0], 0] = (self.alpha/(x[in_range_index[:, 0], 0]** (self.alpha+1))) * (self.xm ** self.alpha)

        return pdf

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the  ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The derivative of the probability distribution (Cx1)
        """
        derivatives_pdf = np.zeros((len(x), 1))
        in_range_index = x >= self.xm
        pdf[in_range_index[:, 0], 0] = -((self.alpha**2+self.alpha) * (self.xm ** self.alpha))/\
                                       (x[in_range_index[:, 0], 0]** (self.alpha+2))
        return derivatives_pdf

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log probablity of ---- distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability of the log probability of ---- distribution (Cx1)
        """
        log_pdf = np.ones((len(x), 1)) * -np.inf
        in_range_index = x >= self.xm
        log_pdf[in_range_index[:, 0], 0] = np.log(self.alpha * (self.xm ** self.alpha)) -\
                                           (self.alpha+1)*np.log(x[in_range_index[:, 0], 0])
        return log_pdf

    def log_pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the log of the ---- distribution
        :param x: An input array of the probability distribution function(Cx1)
        :return: The log probability of the log probability of the occurrence of an independent variable (Cx1)
        """
        derivatives_log_pdf = np.ones((len(x), 1)) * -np.inf
        in_range_index = x >= self.xm
        derivatives_log_pdf[in_range_index[:, 0], 0] = -(self.alpha+1)/x[in_range_index[:, 0], 0]
        return derivatives_log_pdf

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function of ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function of ---- distribution (Cx1)
        """
        cdf = np.zeros((len(x), 1))
        in_range_index = x >= self.xm
        cdf[in_range_index[:, 0], 0] = 1 - (self.xm /x[in_range_index[:, 0], 0]) ** self.alpha
        return cdf


class ExModifiedGaussian(ContinuousDistributions):
    def __init__(self, sigma: float = None, mu: float = None, Lambda: float = None) -> None:
        """

        :param sigma:
        :param mu:
        :param Lambda:
        """
        super(ExModifiedGaussian, self).__init__(sigma=sigma, mu=mu, Lambda=Lambda)


        if self.Lambda <= 0 :
            raise Exception('The value of lambda should be positive (Exponentially modified Gaussian distribution)!')

        if self.sigma <= 0 :
            raise Exception('The value of sigma should be positive (Exponentially modified Gaussian distribution)!')

        if self.variance <= 0 :
            raise Exception('The value of variance should be positive (Exponentially modified Gaussian distribution)!')

        self.Erfc = erfc_fcn
        self.LogErfc = log_erfc

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the probability of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1)
        """
        arg_exp = 2*self.mu+self.Lambda*(self.sigma**2)-2*x
        arg_erfc = (self.mu+self.Lambda*(self.sigma**2)-x)/(self.sigma*np.sqrt(2))
        erfc_val, erfc_diff_val = self.Erfc(arg_erfc)
        pdf = 0.5 * self.Lambda * np.exp(0.5 * self.Lambda * arg_exp) * erfc_val
        return pdf

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the  ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The derivative of the probability distribution (Cx1)
        """
        arg_exp = 2 * self.mu + self.Lambda * (self.sigma ** 2) - 2 * x
        arg_erfc = (self.mu + self.Lambda * (self.sigma ** 2) - x) / (self.sigma * np.sqrt(2))
        erfc_val, erfc_diff_val = self.Erfc(arg_erfc)

        derivatives_pdf = -0.5 * (self.Lambda**2) * np.exp(0.5 * self.Lambda * arg_exp) * erfc_val +\
                          0.5 * self.Lambda *np.exp(0.5 * self.Lambda * arg_exp) * (-1 / (self.sigma * np.sqrt(2))) *\
                          erfc_diff_val
        return derivatives_pdf

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log probablity of ---- distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability of the log probability of ---- distribution (Cx1)
        """

        arg_erfc = (self.mu + self.Lambda * (self.sigma ** 2) - x) / (self.sigma * np.sqrt(2))
        erfc_val, _ = self.Erfc(arg_erfc)

        log_pdf = np.log(0.5*self.Lambda) + 0.5*self.Lambda*(2 * self.mu + self.Lambda * (self.sigma ** 2) - 2 * x) +\
        np.log(erfc_val)
        return log_pdf

    def log_pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the log of the ---- distribution
        :param x: An input array of the probability distribution function(Cx1)
        :return: The log probability of the log probability of the occurrence of an independent variable (Cx1)
        """
        arg_erfc = (self.mu + self.Lambda * (self.sigma ** 2) - x) / (self.sigma * np.sqrt(2))
        erfc_val, erfc_val_diff = self.Erfc(arg_erfc)
        derivatives_log_pdf = self.Lambda * x - (1/(self.sigma*np.sqrt(2))) * (erfc_val_diff/erfc_val)
        return derivatives_log_pdf

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function of ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function of ---- distribution (Cx1)
        """

        arg_exp = 2 * self.mu + self.Lambda * (self.sigma ** 2) - 2 * x
        arg_erfc = (self.mu + self.Lambda * (self.sigma ** 2) - x) / (self.sigma * np.sqrt(2))
        erfc_val, _ = self.Erfc(arg_erfc)
        z = (x - self.mu) / (self.sigma * np.sqrt(2))
        erf_value, _ = self.Erf(z)
        cdf = erf_value - 0.5 * np.exp(0.5 * self.Lambda * arg_exp)*erfc_val
        return cdf


class Triangular(ContinuousDistributions):
    def __init__(self, a:float = None, b:float = None, c:float = None) -> None:
        """

        :param a: left hand side location
        :param b: upper limit of the triangle
        :param c: the mode of the triangle
        """
        super(Triangular, self).__init__(a=a, b=b, c=c)

        if self.a >= self.b:
            raise Exception('The value of a should be lower than b (Triangular distribution)!')

        if self.a > self.c:
            raise Exception('The value of a should be lower/equal than c (Triangular distribution)!')

        if self.c > self.b:
            raise Exception('The value of b should be higher/equal than c (Triangular distribution)!')

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the probability of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1)
        """

        x = np.clip(a=x, a_min=self.a, a_max=self.b)
        pdf = np.zeros((len(x), 1))
        left_index = (self.a <= x) & (x <= self.c)
        right_index = (self.c < x) & (x <= self.b)
        pdf[left_index[:, 0], 0] = 2 * ((x[[left_index[:, 0], 0], 0] - self.a)/((self.b-self.a)*(self.c-self.a)))
        pdf[right_index[:, 0], 0] = 2 * ((self.b-x[[right_index[:, 0], 0], 0]) / ((self.b - self.a) * (self.b - self.c)))
        return pdf

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the  ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The derivative of the probability distribution (Cx1)
        """
        x = np.clip(a=x, a_min=self.a, a_max=self.b)
        left_index = (self.a <= x) & (x <= self.c)
        right_index = (self.c < x) & (x <= self.b)
        derivatives_pdf = np.zeros((len(x), 1))
        derivatives_pdf[left_index[:, 0], 0] = 2 / ((self.b - self.a) * (self.c - self.a))
        derivatives_pdf[right_index[:, 0], 0] = -2 / ((self.b - self.a) * (self.b - self.c))
        return derivatives_pdf

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log probablity of ---- distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability of the log probability of ---- distribution (Cx1)
        """
        x = np.clip(a=x, a_min=self.a, a_max=self.b)
        left_index = (self.a <= x) & (x <= self.c)
        right_index = (self.c < x) & (x <= self.b)
        log_pdf = np.ones((len(x), 1)) * -np.inf

        log_pdf[left_index[:, 0], 0] = np.log(2) + np.log((x[[left_index[:, 0], 0], 0] - self.a)) - np.log(self.b - self.a) - np.log(self.c - self.a)
        log_pdf[right_index[:, 0], 0] = np.log(2) + np.log(self.b - x[[right_index[:, 0], 0], 0]) - np.log(self.b - self.a) - np.log(self.b - self.c)
        return log_pdf

    def log_pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the log of the ---- distribution
        :param x: An input array of the probability distribution function(Cx1)
        :return: The log probability of the log probability of the occurrence of an independent variable (Cx1)
        """
        x = np.clip(a=x, a_min=self.a, a_max=self.b)
        left_index = (self.a <= x) & (x <= self.c)
        right_index = (self.c < x) & (x <= self.b)
        derivatives_log_pdf = np.ones((len(x), 1)) * -np.inf
        derivatives_log_pdf[left_index[:, 0], 0] = 1/(x[[left_index[:, 0], 0], 0] - self.a)
        derivatives_log_pdf[right_index[:, 0], 0] = -1/(self.b - x[[right_index[:, 0], 0], 0])
        return derivatives_log_pdf

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function of ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function of ---- distribution (Cx1)
        """
        x = np.clip(a=x, a_min=self.a, a_max=self.b)
        left_index = (self.a <= x) & (x <= self.c)
        right_index = (self.c < x) & (x <= self.b)
        the_most_right_index = x > self.b
        cdf = np.zeros((len(x), 1))
        cdf[left_index[:, 0], 0] = ((x[[left_index[:, 0], 0], 0] - self.a)**2)/((self.b - self.a)*(self.c - self.a))
        cdf[right_index[:, 0], 0] = 1- ((self.b - x[[right_index[:, 0], 0], 0])**2)/((self.b - self.a)*(self.b - self.c))
        cdf[the_most_right_index[:, 0], 0] = 1.0
        return cdf


class Gumbel (ContinuousDistributions):
    def __init__(self, mu: float = None, beta: float = None) -> None:
        super(Gumbel , self).__init__(mu=mu, beta=beta)

        if self.beta <= 0:
            raise Exception('The value of beta should be positive (Gumbel function)!')

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the probability of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1)
        """
        z = (x - self.mu)/self.beta
        pdf = (1/self.beta) * np.exp(-(z+np.exp(-z)))
        return pdf

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the  ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The derivative of the probability distribution (Cx1)
        """

        z = (x - self.mu) / self.beta
        derivatives_pdf = (1 / self.beta**2)*(np.exp(-z)-1)*np.exp(-(z+np.exp(-z)))
        return derivatives_pdf

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log probablity of ---- distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability of the log probability of ---- distribution (Cx1)
        """
        z = (x - self.mu) / self.beta
        log_pdf = -np.log(self.beta) -z-np.exp(-z)
        return log_pdf

    def log_pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the log of the ---- distribution
        :param x: An input array of the probability distribution function(Cx1)
        :return: The log probability of the log probability of the occurrence of an independent variable (Cx1)
        """
        z = (x - self.mu) / self.beta
        derivatives_log_pdf = (1/self.beta) * (np.exp(-z)-1)
        return derivatives_log_pdf

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function of ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function of ---- distribution (Cx1)
        """
        cdf = np.exp(-np.exp(-(x-self.mu)/ self.beta))
        return cdf



class Logistic(ContinuousDistributions):
    def __init__(self, mu: float = None, sigma: float = None, variance: float = None) -> None:
        """

        :param mu:
        :param sigma:
        :param variance:
        """
        super(Logistic, self).__init__(mu=mu, sigma=sigma, variance=variance)
        if self.sigma <= 0:
            raise Exception('The value of sigma should be positive (Logistic distribution)!')

        if self.variance <= 0:
            raise Exception('The value of variance should be positive (Logistic distribution)!')

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the probability of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1)
        """
        z = (x-self.mu)/self.sigma
        exp_term = np.exp(-z)
        pdf = exp_term /(self.sigma*(1 + exp_term)**2)
        return pdf

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the  ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The derivative of the probability distribution (Cx1)
        """
        z = (x - self.mu) / self.sigma
        exp_term = np.exp(-z)
        ddx_exp_term = (-1/self.sigma)*exp_term
        derivatives_pdf = ((-1/self.sigma**2) * exp_term) * ((1-exp_term**2-2*exp_term)/((1+exp_term)**4))
        return derivatives_pdf

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log probablity of ---- distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability of the log probability of ---- distribution (Cx1)
        """
        z = (x - self.mu) / self.sigma
        log_pdf = -np.log(self.sigma) -z - 2*np.log(1+np.exp(-z))
        return log_pdf

    def log_pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the log of the ---- distribution
        :param x: An input array of the probability distribution function(Cx1)
        :return: The log probability of the log probability of the occurrence of an independent variable (Cx1)
        """
        z = (x - self.mu) / self.sigma
        exp_term = np.exp(-z)
        derivatives_log_pdf = (1/self.sigma)*(-1 + (2*exp_term)/(1+exp_term))
        return derivatives_log_pdf

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function of ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function of ---- distribution (Cx1)
        """
        cdf = 1/(1+np.exp(-(x - self.mu) / self.sigma))
        return cdf


class LogitNormal(ContinuousDistributions):
    def __init__(self, mu: float = None, sigma: float = None, variance: float = None) -> None:
        """

        :param mu:
        :param sigma:
        :param variance:
        """
        super(LogitNormal, self).__init__(mu=mu, sigma=sigma, variance=variance)
        if self.sigma <= 0:
            raise Exception('The value of sigma should be positive (Logistic distribution)!')

        if self.variance <= 0:
            raise Exception('The value of variance should be positive (Logistic distribution)!')

        self.Erf = erf_fcn
    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the probability of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=1-np.finfo(float).eps)
        logit = np.log(x/(1-x))
        pdf = (1/(self.sigma*np.sqrt(2*np.pi))) * (1/(x*(1-x))) * np.exp((-0.5/self.sigma**2)*(logit-self.mu)**2)
        return pdf

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the  ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The derivative of the probability distribution (Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=1 - np.finfo(float).eps)
        logit = np.log(x / (1 - x))
        coef = (1/(self.sigma*np.sqrt(2*np.pi))) * np.exp((-0.5/self.sigma**2) * (logit-self.mu)**2) * (1/((x**2)*(1-x)**2))
        derivatives_pdf = coef * ((2*x-1) - (1/self.sigma**2) * (logit-self.mu))
        return derivatives_pdf

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log probablity of ---- distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability of the log probability of ---- distribution (Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=1 - np.finfo(float).eps)
        logit = np.log(x / (1 - x))
        log_pdf = -np.log(self.sigma*np.sqrt(2*np.pi)) - np.log(x*(1-x)) - ((0.5/self.sigma**2) * (logit-self.mu)**2)
        return log_pdf

    def log_pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the log of the ---- distribution
        :param x: An input array of the probability distribution function(Cx1)
        :return: The log probability of the log probability of the occurrence of an independent variable (Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=1 - np.finfo(float).eps)
        logit = np.log(x / (1 - x))
        derivatives_log_pdf = (1/(x-x**2))*((2*x-1) - (1/self.sigma**2)*(logit-self.mu))
        return derivatives_log_pdf

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function of ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function of ---- distribution (Cx1)
        """
        x = np.clip(a=x, a_min=np.finfo(float).eps, a_max=1 - np.finfo(float).eps)
        logit = np.log(x / (1 - x))
        input_argument = (logit - self.mu)/(self.sigma*np.sqrt(2))
        cdf = 0.5 * (1+ self.Erf(input_argument))
        return cdf



#######################################################################################################################
########################################################################################################################
#######################################################################################################################

class MyClass(ContinuousDistributions):
    def __init__(self) -> None:
        super(MyClass, self).__init__()


    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the probability of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1)
        """
        pdf = np.zeros((len(x), 1))

        return pdf

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the  ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The derivative of the probability distribution (Cx1)
        """
        derivatives_pdf = np.zeros((len(x), 1))
        return derivatives_pdf

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the log probablity of ---- distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability of the log probability of ---- distribution (Cx1)
        """
        log_pdf = np.ones((len(x), 1)) * -np.inf

        return log_pdf

    def log_pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the log of the ---- distribution
        :param x: An input array of the probability distribution function(Cx1)
        :return: The log probability of the log probability of the occurrence of an independent variable (Cx1)
        """
        derivatives_log_pdf = np.ones((len(x), 1)) * -np.inf
        return derivatives_log_pdf

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the cumulative distribution function of ---- distribution
        :param x: An array of the input variable (Cx1)
        :return: The cumulative distribution function of ---- distribution (Cx1)
        """
        cdf = np.zeros((len(x), 1))
        return cdf



###################################################################################################
x = (np.random.uniform(low = -200, high=200, size=10000)).reshape((-1,1))
ts = Normal(sigma=4,mu=7)
pdf_val = ts.pdf(x)
der_pdf =ts.pdf_diff(x)
log_pdf_val =ts.log_prob(x)
der_log_ = ts.log_prob_diff(x)
cdf_val = ts.cdf(x)
samples = ts.sample(size=50000)

if any(np.isnan(pdf_val)):
    print('there is NAN in pdf')

if any(np.isnan(der_pdf)):
    print('there is NAN in diffpdf')

if any(np.isnan(log_pdf_val)):
    print('there is NAN in log pdf')

if any(np.isnan(der_log_)):
    print('there is NAN in diff log pdf')

if any(np.isnan(cdf_val)):
    print('there is NAN in cdf')

if any(np.isnan(samples)):
    print('there is NAN in samples')



fig, ((ax1, ax2,ax3),  (ax4,ax5,ax6)) = plt.subplots(2, 3)

ax1.plot(x,pdf_val,'*')
ax1.set_title('PDF values')

ax4.plot(x,log_pdf_val,'*')
ax4.set_title('derivatives of PDF ')

ax2.plot(x,log_pdf_val,'*')
ax2.set_title('LOG of PDF values')

ax5.plot(x,der_log_,'*')
ax5.set_title('derivatives of LOG of PDF ')

ax3.plot(x,cdf_val,'*')
ax3.set_title('CDF ')

ax6.hist(samples,bins=20)
ax6.set_title('samples ')

show()

x

