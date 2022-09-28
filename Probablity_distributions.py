import numpy as np
from matplotlib.pyplot import plot, show

# =============================================================================
#
# =============================================================================
class Continuous_Uniform:
    def __init__(self, lb: float = 0.0, ub: float = 1.0):
        """
        The continuous uniform distribution
        :param lb: the lower bound of the uniform distribution
        :param ub: the upper bound of the uniform distribution
        """
        self.lb = lb
        self.ub = ub
    def Prob(self, x: float = 0.5):
        if x <= self.lb or x >= self.ub:
            return 0
        else:
            return 1 / (self.ub - self.lb)

    def Log_prob(self, x: float = 0.5):
        if x <= self.lb or x >= self.ub:
            return -np.inf
        else:
            return -np.log(self.ub - self.lb)

    def Visualize(self, lower_lim: float = -10, upper_lim: float = -10):
        X = np.linspace(lower_lim, upper_lim, 1000)
        Y = list()
        for i in range(len(X)):
            Y.append(self.Prob(X[i]))
        plot(list(X.ravel()), Y)
        show()


# =============================================================================
#
# =============================================================================
class Continuous_Gaussian:
    def __init__(self, mu: float = 0.0, std: float = 1.0):
        self.mu = mu
        self.std = std

    def Prob(self, x: float = 0.5):
        return (1 / (self.std * np.sqrt(2 * np.pi))) * np.exp(-((x - self.mu) ** 2) / (2 * self.std ** 2))

    def Log_prob(self, x: float = 0.5):
        return -np.log(self.std * np.sqrt(2 * np.pi)) - ((x - self.mu) ** 2) / (2 * self.std ** 2)

    def Visualize(self, lower_lim: float = -10, upper_lim: float = -10):
        X = np.linspace(lower_lim, upper_lim, 1000)
        Y = list()
        for i in range(len(X)):
            Y.append(self.Prob(X[i]))
        plot(list(X.ravel()), Y)


# =============================================================================
#
# =============================================================================
class Continuous_Truncated_Gaussian:
    def __init__(self, mu: float = 0.0, std: float = 1.0, lb: float = -1, ub: float = 1):
        self.mu = mu
        self.std = std
        self.lb = lb
        self.ub = ub

    def Erf(self, z):
        return (2 / (np.sqrt(np.pi))) * (z - (z ** 3 / 3) + (z ** 5 / 10) - (z ** 7 / 42) + (z ** 9 / 216))

    def Prob(self, x):
        if x <= self.lb or x >= self.ub:
            return 0
        else:
            L1 = (self.ub - self.mu) / self.std
            L2 = (self.lb - self.mu) / self.std
            L = (x - self.mu) / self.std
            Fi_1 = 0.5 * (1 + self.Erf(L1 / 2 ** 0.5))
            Fi_2 = 0.5 * (1 + self.Erf(L2 / 2 ** 0.5))
            fi = (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * L ** 2)
        return (1 / self.std) * (fi / (Fi_1 - Fi_2))

    def Log_prob(self, x: float = 0.5):
        if x <= self.lb or x >= self.ub:
            return -np.inf
        else:
            L1 = (self.ub - self.mu) / self.std
            L2 = (self.lb - self.mu) / self.std
            L = (x - self.mu) / self.std
            Fi_1 = 0.5 * (1 + self.Erf(L1 / 2 ** 0.5))
            Fi_2 = 0.5 * (1 + self.Erf(L2 / 2 ** 0.5))
            return -np.log(self.std) - np.log(Fi_1 - Fi_2) - np.log((np.sqrt(2 * np.pi))) - 0.5 * L ** 2

    def Visualize(self, lower_lim: float = -10, upper_lim: float = -10):
        X = np.linspace(lower_lim, upper_lim, 1000)
        Y = list()
        for i in range(len(X)):
            Y.append(self.Prob(X[i]))
        plot(list(X.ravel()), Y)


# =============================================================================
#
# =============================================================================
class Continuous_Half_Gaussian:
    def __init__(self, std: float = 1.0):
        self.std = std

    def Prob(self, x: float = 0.5):
        if x <= 0:
            return 0
        else:
            return (np.sqrt(2) / (self.std * np.sqrt(np.pi))) * np.exp(-((x - self.mu) ** 2) / (2 * self.std ** 2))

    def Log_prob(self, x: float = 0.5):
        if x <= 0:
            return -np.inf
        else:
            return np.log(np.sqrt(2) / (self.std * np.sqrt(np.pi))) - (x ** 2) / (2 * self.std ** 2)

    def Visualize(self, lower_lim: float = -10, upper_lim: float = -10):
        X = np.linspace(lower_lim, upper_lim, 1000)
        Y = list()
        for i in range(len(X)):
            Y.append(self.Prob(X[i]))
        plot(list(X.ravel()), Y)
# =============================================================================
#
# =============================================================================




