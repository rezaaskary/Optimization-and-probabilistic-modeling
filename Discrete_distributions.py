import numpy as np
from scipy.special import factorial
from matplotlib.pyplot import plot, grid, figure, show


class DiscreteDistributions:
    def __init__(self,
                 n: int = None,
                 p: int = None) -> None:

        if isinstance(n, int):
            self.n = n
        elif n is None:
            self.n = None
        else:
            raise Exception('The lower bound is not specified correctly!')

        if isinstance(p, int):
            self.p = p
        elif p is None:
            self.p = None
        else:
            raise Exception('The lower bound is not specified correctly!')

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


class Binomial(DiscreteDistributions):
    def __init__(self, n: int = None, p: int = None) -> None:
        super(Binomial, self).__init__(n=n, p=p)

    @property
    def statistics(self):
        """
        Statistics calculated for the ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    def pdf(self, x: int) -> np.ndarray:
        """
        Parallelized calculating the probability of the ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1)
        """
        pdf = (factorial(self.n) / (factorial(self.n - self.x) * factorial(self.x))) * (self.p ** x) * \
              ((1 - self.p) ** (self.n - x))
        return pdf

    def pdf_diff(self, x: np.ndarray) -> np.ndarray:
        """
        Parallelized calculating the derivatives of the  ----- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The derivative of the probability distribution (Cx1)
        """
        derivatives_pdf = np.zeros((len(x), 1))
        return derivatives_pdf

    def log_pdf(self, x: int) -> np.ndarray:
        """
        Parallelized calculating the log probablity of ---- distribution
        :param x: An integer array determining the variable we are calculating its probability distribution (Cx1)
        :return: The log probability of the log probability of ---- distribution (Cx1)
        """

        log_pdf = np.log((factorial(self.n) / (factorial(self.n - self.x) * factorial(self.x)))) + \
                  x * np.log(self.p) + (self.n - self.x) * np.log(1 - self.p)
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
