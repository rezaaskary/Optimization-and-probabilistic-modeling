import numpy as np


def lower_incomplete_gamma_fcn(s, x, method: str = 'numerical'):
    """
    calculating the lower incomplete Gamma function used for calculating various pdf
    :param s:
    :param x:
    :param method:
    :return:
    """
    if method == 'numerical':
        t = np.linspace(0, x, 10000)
        f = np.exp(-t) * t ** (s - 1)
        deltat = t[1] - t[0]
        gamma_value = deltat * (f[1:-1]).sum() + 0.5 * deltat * (f[0] + f[-1])
    return gamma_value

def arctan_fcn(z, method: str = 'taylor'):
    if method == 'taylor':
        N = 10
        if z >= 1:
            fcn_value = 0.5 * np.pi
            derivative_value = 0
            for n in range(N):
                denom_value = ((2*n+1)*(z**(2*n+1)))
                fcn_value += ((-1)**(n+1)) / denom_value
                derivative_value += (((-1)**n) * (2*n+1) * (2*n+1) * z**(2*n)) /denom_value**2
        elif z <= -1:
            fcn_value = -0.5 * np.pi
            derivative_value = 0
            for n in range(N):
                denom_value = ((2 * n + 1) * (z ** (2 * n + 1)))
                fcn_value += ((-1) ** (n + 1)) / denom_value
                derivative_value += (((-1) ** n) * (2 * n + 1) * (2 * n + 1) * z ** (2*n)) / denom_value ** 2
        else:
            fcn_value = 0
            derivative_value = 0
            for n in range(N):
                denom_value = ((2 * n + 1) * (z ** (2 * n + 1)))
                fcn_value += (((-1) ** n) * (z ** (2 * n + 1)) ) / (2 * n + 1)
                derivative_value += ((-1) ** n) * z ** (2 * n)

    return fcn_value, derivative_value

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

def erf_fcn(z, method: str = 'fast', terms:int = 20)->(np.ndarray, np.ndarray):
    """
    The error function used to calculate the truncated gaussian distribution
    :param z: normalized input variable
    :return: the value of the error function
    """
    derivatives_Erf = (2 / np.sqrt(np.pi)) * np.exp(-z ** 2)
    if method == 'fast':
        Erf_function_value = ((2 / (np.sqrt(np.pi))) * (z - (z ** 3 / 3) + (z ** 5 / 10) - (z ** 7 / 42) +
                                                        (z ** 9 / 216) - (z ** 11 / 1320) + (z ** 13 / 9360) +
                                                        (z ** 15 / 75600)))
    elif method == 'numerical':
        t = np.linspace(0, z, 10000)
        f = (2/np.sqrt(np.pi)) * np.exp(-t**2)
        deltat = t[1] - t[0]
        Erf_function_value = deltat * (f[1:-1]).sum() + 0.5 * deltat * (f[0] + f[-1])

    elif method == 'taylor':
        erf_val = 0
        for n in range(terms):
            erf_val += (((-1)**n) * (z**(2 * n + 1))) / (np.math.factorial(n) * (2*n+1))
        Erf_function_value = erf_val * (2/np.sqrt(np.pi))
    else:
        raise Exception('The method for calculating the Error function is not specified correctly!')
    return Erf_function_value, derivatives_Erf


def erfc_fcn(z, method: str = 'fast', terms:int = 20)->(np.ndarray, np.ndarray):
    derivatives_erfc = - (2 / np.sqrt(np.pi)) * np.exp(-z ** 2)
    if method == 'fast':
        Erf_function_value = ((2 / (np.sqrt(np.pi))) * (z - (z ** 3 / 3) + (z ** 5 / 10) - (z ** 7 / 42) +
                                                        (z ** 9 / 216) - (z ** 11 / 1320) + (z ** 13 / 9360) +
                                                        (z ** 15 / 75600)))
    elif method == 'numerical':
        t = np.linspace(0, z, 10000)
        f = (2/np.sqrt(np.pi)) * np.exp(-t**2)
        deltat = t[1] - t[0]
        Erf_function_value = deltat * (f[1:-1]).sum() + 0.5 * deltat * (f[0] + f[-1])

    elif method == 'taylor':
        erf_val = 0
        for n in range(terms):
            erf_val += (((-1)**n) * (z**(2 * n + 1))) / (np.math.factorial(n) * (2*n+1))
        Erf_function_value = erf_val * (2/np.sqrt(np.pi))
    else:
        raise Exception('The method for calculating the Error function is not specified correctly!')
    return 1 - Erf_function_value, derivatives_erfc

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

