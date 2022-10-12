import numpy as np


class ErfFcn(object):
    def __init__(self, method: str = 'simpson', intervals: int = 10000):
        """

        :param method:
        :param intervals:
        :return:
        """
        if isinstance(method, str):
            self.method = method
        else:
            raise Exception('Please correctly enter the method of integration (Error function)!')

        if isinstance(intervals, int):
            self.intervals = intervals
        else:
            raise Exception('Please correctly enter the number of intervals (Error function)!')

        if self.method == 'simpson':
            self.fcn_value = self.simpson
        elif self.method == 'trap':
            self.fcn_value = self.trapezoidal
        else:
            raise Exception('The entered methode is unknown (Error function)!')

    def trapezoidal(self,z: np.ndarray = None)->np.ndarray:
        """

        :param z:
        :return:
        """

        t = (np.linspace(0, z, self.intervals))[:, :, 0]
        f = (2 / np.sqrt(np.pi)) * np.exp(-t ** 2)
        deltat = t[1:2, :] - t[0:1, :]
        return deltat * (f[1:-1, :]).sum(axis=0) + 0.5 * deltat * (f[0:1, :] + f[-1:, :])

    def simpson(self, z: np.ndarray = None) -> np.ndarray:
        """

        :param z:
        :return:
        """
        t = (np.linspace(0, z, self.intervals))[:, :, 0]
        f = (2 / np.sqrt(np.pi)) * np.exp(-t ** 2)
        t_m = 0.5 * (t[1:, :] + t[:-1, :])
        f_m = (2 / np.sqrt(np.pi)) * np.exp(-t_m ** 2)
        deltat = t[1:2, :] - t[0:1, :]
        return (deltat / 6) * (f[0:1, :] + f[-1:, :]) + (deltat / 3) * (f[1:-1, :]).sum(axis=0) + (
                    deltat * (2 / 3)) * f_m.sum(axis=0)

    def derivatives(self, z: np.ndarray = None) -> np.ndarray:
        """
        :param z:
        :return:
        """
        return (2 / np.sqrt(np.pi)) * np.exp(-z ** 2)


class ErfcFcn:
    def __init__(self, method: str = 'simpson', intervals: int = 10000):
        if isinstance(method, str):
            self.method = method
        else:
            raise Exception('Please correctly enter the method of integration (Complementary Error function)!')

        if isinstance(method, int):
            self.intervals = intervals
        else:
            raise Exception('Please correctly enter the number of intervals (Complementary Error function)!')

        if self.method == 'simpson':
            self.fcn_value = self.simpson
        elif self.method == 'trap':
            self.fcn_value = self.trapezoidal
        else:
            raise Exception('The entered methode is unknown (Complementary Error function)!')

    def trapezoidal(self,z: np.ndarray = None)->np.ndarray:
        """

        :param z:
        :return:
        """

        t = (np.linspace(0, z, self.intervals))[:, :, 0]
        f = (2 / np.sqrt(np.pi)) * np.exp(-t ** 2)
        deltat = t[1:2, :] - t[0:1, :]
        integral = deltat * (f[1:-1, :]).sum(axis=0) + 0.5 * deltat * (f[0:1, :] + f[-1:, :])
        return 1-integral

    def simpson(self, z: np.ndarray = None) -> np.ndarray:
        """

        :param z:
        :return:
        """
        t = (np.linspace(0, z, self.intervals))[:, :, 0]
        f = (2 / np.sqrt(np.pi)) * np.exp(-t ** 2)
        t_m = 0.5 * (t[1:, :] + t[:-1, :])
        f_m = (2 / np.sqrt(np.pi)) * np.exp(-t_m ** 2)
        deltat = t[1:2, :] - t[0:1, :]
        integral = (deltat / 6) * (f[0:1, :] + f[-1:, :]) + (deltat / 3) * (f[1:-1, :]).sum(axis=0) + (
                    deltat * (2 / 3)) * f_m.sum(axis=0)
        return 1-integral

    def derivatives(self, z: np.ndarray = None) -> np.ndarray:
        """

        :param z:
        :return:
        """
        return -(2 / np.sqrt(np.pi)) * np.exp(-z ** 2)


class LowerIncompleteGamma:
    def __init__(self, method: str = 'simpson', intervals: int = 10000):
        """

        :param method:
        :param intervals:
        :return:
        """

        if isinstance(method, str):
            self.method = method
        else:
            raise Exception('Please correctly enter the method of integration (Complementary Error function)!')

        if isinstance(method, int):
            self.intervals = intervals
        else:
            raise Exception('Please correctly enter the number of intervals (Complementary Error function)!')

        if self.method == 'simpson':
            self.fcn_value = self.simpson
        elif self.method == 'trap':
            self.fcn_value = self.trapezoidal
        else:
            raise Exception('The entered methode is unknown (Complementary Error function)!')

    def trapezoidal(self, z: np.ndarray = None, s: np.ndarray = None) -> np.ndarray:
        """

        :param z:
        :param s:
        :return:
        """
        t = (np.linspace(0, z, self.intervals))[:, :, 0]
        f = np.exp(-t) * t ** (s - 1)
        deltat = t[1:2, :] - t[0:1, :]
        integral = deltat * (f[1:-1, :]).sum(axis=0) + 0.5 * deltat * (f[0:1, :] + f[-1:, :])

        return integral

    def simpson(self, z: np.ndarray = None, s: np.ndarray = None) -> np.ndarray:
        """

        :param z:
        :param s:
        :return:
        """
        t = (np.linspace(0, z, self.intervals))[:, :, 0]
        f = np.exp(-t) * t ** (s - 1)
        t_m = 0.5 * (t[1:, :] + t[:-1, :])
        f_m = np.exp(-t_m) * t_m ** (s - 1)
        deltat = t[1:2, :] - t[0:1, :]
        integral = (deltat / 6) * (f[0:1, :] + f[-1:, :]) + (deltat / 3) * (f[1:-1, :]).sum(axis=0) + (
                deltat * (2 / 3)) * f_m.sum(axis=0)
        return integral


class ArcTangent:
    def __init__(self, terms: int = 10):

        if isinstance(terms, int):
            self.terms = terms
        else:
            raise Exception('Please correctly enter the number of terms (ArcTangent function)!')

    def fcn_value(self, z: np.ndarray = None) -> np.ndarray:
        Lindex = z <= -1
        Rindex = z >= 1
        Mindex = (z > -1) & (z < 1)

        fcn_value = np.zeros(len(z), 1)
        fcn_value[Rindex[:, 0], 0] = 0.5 * np.pi
        fcn_value[Lindex[:, 0], 0] = -0.5 * np.pi

        for n in range(self.terms):
            fcn_value[Rindex[:, 0], 0] += ((-1) ** (n + 1)) / ((2 * n + 1) * (z[Rindex[:, 0], 0] ** (2 * n + 1)))
            fcn_value[Mindex[:, 0], 0] += ((-1) ** (n + 1)) / ((2 * n + 1) * (z[Mindex[:, 0], 0] ** (2 * n + 1)))
            fcn_value[Lindex[:, 0], 0] += ((-1) ** (n + 1)) / ((2 * n + 1) * (z[Lindex[:, 0], 0] ** (2 * n + 1)))
        return fcn_value

    def derivatives(self, z: np.ndarray = None) -> np.ndarray:
        Lindex = z <= -1
        Rindex = z >= 1
        Mindex = (z > -1) & (z < 1)
        derivative_value = np.zeros(len(z), 1)
        for n in range(self.terms):
            derivative_value[Rindex[:, 0], 0] += (((-1)**n) * (2*n+1) * (2*n+1) * ([Rindex[:, 0], 0])**(2*n)) /\
                                                 ((2*n+1)*((z[Rindex[:, 0], 0])**(2*n+1)))**2
            derivative_value[Mindex[:, 0], 0] += ((-1) ** n) * ([Mindex[:, 0], 0]) ** (2 * n)
            derivative_value[Lindex[:, 0], 0] += (((-1) ** n) * (2 * n + 1) * (2 * n + 1) *
                                                  (z[Lindex[:, 0], 0]) ** (2*n)) / ((2 * n + 1) *
                                                                            ((z[Lindex[:, 0], 0]) ** (2 * n + 1)))**2
        return derivative_value


class GammaFcn:
    def __int__(self, method: str = 'simpson', intervals: int = 10000):
        """

        :param method:
        :param intervals:
        :return:
        """

        if isinstance(method, str):
            self.method = method
        else:
            raise Exception('Please correctly enter the method of integration (Complementary Error function)!')

        if isinstance(method, int):
            self.intervals = intervals
        else:
            raise Exception('Please correctly enter the number of intervals (Complementary Error function)!')

        if self.method == 'simpson':
            self.fcn_value = self.simpson
        elif self.method == 'trap':
            self.fcn_value = self.trapezoidal
        else:
            raise Exception('The entered methode is unknown (Complementary Error function)!')

    def trapezoidal(self, s: np.ndarray = None) -> np.ndarray:
        """

        :param z:
        :param s:
        :return:
        """
        t = np.linspace(np.finfo(float).eps, 100, self.intervals)
        f = np.exp(-t) * t ** (s - 1)
        deltat = t[1:2] - t[0:1]
        integral = deltat * (f[:, 1:-1]).sum(axis=1) + 0.5 * deltat * (f[:, 0:1] + f[:, -1:])
        return integral

    def simpson(self, z: np.ndarray = None, s: np.ndarray = None) -> np.ndarray:
        """

        :param z:
        :param s:
        :return:
        """

        t = (np.linspace(np.finfo(float).eps, 100, self.intervals))
        f = np.exp(-t) * t ** (s - 1)
        t_m = 0.5 * (t[1:] + t[:-1])
        f_m = np.exp(-t_m) * t_m ** (s - 1)
        deltat = t[1:2] - t[0:1]
        integral = (deltat / 6) * (f[:, 0:1] + f[:, -1:]) + (deltat / 3) * (f[:, 1:-1]).sum(axis=1) + (
                deltat * (2 / 3)) * f_m.sum(axis=1)
        return integral















def gamma_fcn(z, method: str = 'numerical'):
    """
    calcualting the Gamma function for use in other probablity distribution
    :param z:
    :return:
    """

    if method == 'stirling':
        gamma_value = np.sqrt(2 * np.pi * z) * np.power(z, z - 1) * np.exp(-z) * (
                    1 + (1 / (12 * z)) + (1 / (288 * z ** 2)) - (1 / (51480 * z ** 3)))
        return gamma_value

    elif method == 'numerical':
        t = np.linspace(0, 100, 10000)
        f = np.exp(-t) * t ** (z - 1)
        deltat = t[1] - t[0]
        gamma_value = deltat * (f[1:-1]).sum() + 0.5 * deltat * (f[0] + f[-1])
        return gamma_value
    else:
        raise Exception('The method of calculating the Gamma function is not specified correctly!')

def beta_fcn(x, y, method: str = 'numerical'):
    """
    calculates the Beta function B(a,y)
    :param x: a float value such that 0<x<1
    :param y: a float value such that 0<y<1
    :param method: the method used to calcualte the beta function
    :return:
    """
    def beta_kernel(x, y, t):
        return (t ** (x - 1)) * (1 - t)**(y-1)

    if method == 'numerical':
        t = np.linspace(0, 1, 10000)
        f = beta_kernel(x, y, t)
        deltat = t[1] - t[0]
        return deltat * (f[1:-1]).sum() + 0.5 * deltat * (f[0] + f[-1])


def erfinv_fcn(z, method: str = 'fast'):
    """
    The inversion of error function
    :param z:
    :param method:
    :return:
    """
    if method == 'fast':
        erfinv = 0.5 * np.sqrt(np.pi) * (z + (np.pi/12)*(x**3) + (((np.pi**2)*7)/480)*(x**5) +
                                     (((np.pi**3)*127)/40320)*(x**7) +
                                     (((np.pi**4)*4369)/5806080)*(x**9) + (((np.pi**5)*34807)/182476800)*(x**11))
    else:
        raise Exception('The method of calculating the inversion of Erf is not specified correctly!')
    return erfinv


def log_erf(z, method: str = 'fast'):
    Erf_function_value = ((2 / (np.sqrt(np.pi))) * (z - (z ** 3 / 3) + (z ** 5 / 10) - (z ** 7 / 42) +
                                                    (z ** 9 / 216) - (z ** 11 / 1320) + (z ** 13 / 9360) +
                                                    (z ** 15 / 75600)))

    log_erf_value = np.log(2 / (np.sqrt(np.pi))) + np.log((z - (z ** 3 / 3) + (z ** 5 / 10) - (z ** 7 / 42) +
                                                    (z ** 9 / 216) - (z ** 11 / 1320) + (z ** 13 / 9360) +
                                                    (z ** 15 / 75600)))
    derivatives_Erf = (2 / np.sqrt(np.pi)) * np.exp(-z ** 2)
    log_erf_diff = derivatives_Erf / Erf_function_value
    return log_erf_value, log_erf_diff


def log_erfc(z, method: str = 'fast'):
    Erfc_function_value = 1 - ((2 / (np.sqrt(np.pi))) * (z - (z ** 3 / 3) + (z ** 5 / 10) - (z ** 7 / 42) +
                                                    (z ** 9 / 216) - (z ** 11 / 1320) + (z ** 13 / 9360) +
                                                    (z ** 15 / 75600)))
    log_erf_value = np.log(Erfc_function_value)
    derivatives_Erfc = -(2 / np.sqrt(np.pi)) * np.exp(-z ** 2)
    log_erf_diff = derivatives_Erfc / Erfc_function_value
    return log_erf_value, log_erf_diff


def bessel_I_s(z: np.ndarray, s):
    N = 10
    I_s = 0
    for m in range(N):
        I_s += ((0.5*z)**(2*m+s)) / (np.math.factorial(m) * gamma_fcn(m+s+1))

    return I_s


import math

# (np.random.uniform(low=-20, high=20, size=10000)).reshape((-1, 1))
# dd=np.random.default_rng(12345)
# T=dd.uniform(low=0, high=1,size=10).reshape((-1,1))
# V = erf_fcn(T, method='trap',terms=10000)
# V1 = erf_fcn(T,method='simpson',terms=10000)
#
# V2 =math.erf(ff)
# print(V2-V)
# V
