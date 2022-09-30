import numpy as np
from matplotlib.pyplot import plot, show

#=========================================================================================================
class Continuous_Uniform:
    def __init__(self, lb: float = 0.0, ub: float = 1.0):
        """
        The continuous uniform distribution
        :param lb: the lower bound of the uniform distribution
        :param ub: the upper bound of the uniform distribution
        """
        self.lb = lb
        self.ub = ub
    def Prob(self, x: float = 0.5)->np.ndarray:
        """
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: the probablity of the occurance of the given variable
        """
        if x <= self.lb or x >= self.ub:
            return 0
        else:
            return 1 / (self.ub - self.lb)

    def Log_prob(self, x: float = 0.5)->np.ndarray:
        """
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: The log of the probablity distribution of the given variable
        """
        if x <= self.lb or x >= self.ub:
            return -np.inf
        else:
            return -np.log(self.ub - self.lb)

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

class Continuous_Gaussian:
    def __init__(self, mu: float = 0.0, std: float = 1.0):
        """
        The continuous gaussian distribution function
        :param mu: the center of the gaussian distribution
        :param std: the standard deviation of gaussian distribution
        """
        self.mu = mu
        self.std = std

    def Prob(self, x: float = 0.5)->np.ndarray:
        """
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: the probablity of the occurance of the given variable
        """
        return (1 / (self.std * np.sqrt(2 * np.pi))) * np.exp(-((x - self.mu) ** 2) / (2 * self.std ** 2))

    def Log_prob(self, x: float = 0.5)->np.ndarray:
        """
        :param x: an integer value determining the variable we are calculating its probablity distribution
        :return: The log of the probablity distribution of the given variable
        """
        return -np.log(self.std * np.sqrt(2 * np.pi)) - ((x - self.mu) ** 2) / (2 * self.std ** 2)

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
#=========================================================================================================


class Continuous_Truncated_Gaussian:
    def __init__(self, mu: float = 0.0, std: float = 1.0, lb: float = -1, ub: float = 1):
        """
       The continuous uniform distribution
       :param mu: the center bound of the truncated normal distribution
       :param std: the standard deviation bound of the truncated normal distribution
       :param lb: the lower bound of the truncated normal distribution
       :param ub: the upper bound of the truncated normal distribution
       """
        self.mu = mu
        self.std = std
        self.lb = lb
        self.ub = ub

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

    def Log_prob(self, x: float = 0.5)->np.ndarray:
        """
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
#=========================================================================================================

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
#=========================================================================================================




def gaussian_liklihood_single_variable(measured:np.ndarray, estimated: np.ndarray, N:int, sigma: np.ndarray) -> np.ndarray:
    """
    The single variable Gausian liklihood function
    :param measured: The measured variable
    :param estimated: The estimated variable(or calculated from a model)
    :param N: The number of measured samples
    :param sigma: The standard deviation of the error estimation
    :return: the log_liklihood function
    """
    log_gauss = -N*np.log(sigma * np.sqrt(2 * np.pi)) - (((measured - estimated) ** 2) / (2 * sigma ** 2)).sum()
    return log_gauss
    #====================================================================================

def gaussian_liklihood_single_variable_vectorized(measured: np.ndarray, estimated: np.ndarray, N: int,C: int, sigma: np.ndarray) -> np.ndarray:
    """
    The single variable Gausian liklihood function
    :param measured: The measured variable (NxC)
    :param estimated: The estimated variable or calculated from a model (NxC)
    :param N: An integer indicating the number of measured samples
    :param C: An integer indicating the number of Chains
    :param sigma: The standard deviation of the error estimation (1xC)
    :return: A numpy array indicating the log_liklihood function (1xC)
    """
    vectorized_error = ((measured - estimated)**2).sum(axis=0)
    log_gauss = -N*np.log(sigma * np.sqrt(2 * np.pi)) - (vectorized_error / (2 * sigma ** 2))
    return log_gauss
    #====================================================================================

def gaussian_liklihood_multivariable(measured:np.ndarray, estimated:np.ndarray, N:int, Covariance: np.ndarray, K:int) -> np.ndarray:
    """
    The log liklihood of the Multivariable gaussian distribution used for multivariable fitting (multivariables objective function)
    :param measured: KxN measured parameters (K dimentional parameters and N sampling points)
    :param estimated:KxN estimated parameters (K dimentional parameters and N sampling points)
    :param N: An integer indicating the number of measurements
    :param Covariance: A positive definite square matrix indicating the covariance matrix of the multivariable Normal distribution (KxK)
    :param K: The dimention of the multivariable gaussian distribution
    :return: The log liklihood of
    """
    inv_cov = np.linalg.inv(Covariance)     # calcualting the inversion of the covariance matrix
    det_cov = np.linalg.det(Covariance)     # calcualting the determinent of the covariance matrix
    Error = measured - estimated            # KxN error matrix
    log_liklihood_gaussian = -N * np.log(np.sqrt(((2 * np.pi)**K) * det_cov)) - (0.5 * (np.diag(Error.T @ inv_cov @ Error))).sum()  # the log_liklihood gaussian
    return log_liklihood_gaussian


def gaussian_liklihood_multivariable_vectorized(measured:np.ndarray, estimated:np.ndarray, N:int, Covariance: np.ndarray, K:int, C:int, Diagonal:True)->np.ndarray:
    """
    The log liklihood of the Multivariable gaussian distribution used for multivariable fitting (multivariables objective function)
    :param measured: KxNxC measured parameters (K dimentional parameters and N sampling points and C chains)
    :param estimated:KxNxC estimated parameters (K dimentional parameters and N sampling points and C chains)
    :param N: An integer indicating the number of measurements
    :param Covariance: A positive definite square matrix indicating the covariance matrix of the multivariable Normal distribution (KxKxC)
    :param K: The dimention of the multivariable gaussian distribution
    :return: The log liklihood of the multivariable gaussian distribution
    """
    diagonal_indexes = np.arange(K, dtype=int)
    diagonal_indexes_samples = np.arange(N, dtype=int)
    inv_cov = Covariance.copy()   # predefining the inversion matrices
    Error = measured - estimated  # KxNxC error matrix
    Error_T = np.transpose(Error, axes=(1, 0, 2))  # NxKxC error matrix

    if Diagonal:
            # calcualting the inversion of the covariance matrix
        inv_cov[diagonal_indexes, diagonal_indexes, :] = 1 / inv_cov[diagonal_indexes, diagonal_indexes,:]  #KxKxX tensor
        det_cov = np.prod(Covariance[diagonal_indexes, Covariance,:],axis = 0)      # 1xC array

        vectorized_mahalanobis_distance = (((Error_T[:, :, None] * inv_cov).sum(axis = 1))[:, :, None] * T).sum(axis = 1) # NxNxC
        mahalanobis_distance = vectorized_mahalanobis_distance[diagonal_indexes_samples, diagonal_indexes_samples, :].sum(axis = 0)
        log_liklihood = -N * np.log(np.sqrt(((2 * np.pi) ** K) * det_cov)) - 0.5 * mahalanobis_distance
        return log_liklihood

    elif not Diagonal:
        det_cov = np.zeros((1,C))
        for c in range(C):
            det_cov[0, c] = np.linalg.det(Covariance[:, :, c])
            inv_cov[:, :, c:c+1] = np.linalg.inv(Covariance[:, :, c])

        vectorized_mahalanobis_distance = (((Error_T[:, :, None] * inv_cov).sum(axis=1))[:, :, None] * T).sum(axis=1)  # NxNxC
        mahalanobis_distance = vectorized_mahalanobis_distance[diagonal_indexes_samples, diagonal_indexes_samples, :].sum(axis=0)
        log_liklihood = -N * np.log(np.sqrt(((2 * np.pi) ** K) * det_cov)) - 0.5 * mahalanobis_distance
        return log_liklihood
    else:
        raise Exception('The type of calculating the liklihood function is not correctly specified!')

