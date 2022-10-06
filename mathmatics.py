import numpy as np

def Gamma(z, method: str = 'numerical'):
    """
    calcualting the Gamma function for use in other probablity distribution
    :param z:
    :return:
    """
    def gamma_kernel(z, t):
        return np.exp(-t) * t ** (z - 1)

    if method is 'stirling':
        gamma_value = np.sqrt(2 * np.pi * z) * np.power(z, z - 1) * np.exp(-z) * (
                    1 + (1 / (12 * z)) + (1 / (288 * z ** 2)) - (1 / (51480 * z ** 3)))
        return gamma_value

    elif method is 'numerical':
        t = np.linspace(0, 100, 10000)
        f = gamma_kernel(z, t)
        deltat = t[1] - t[0]
        gamma_value = deltat * (f[1:-1]).sum() + 0.5 * deltat * (f[0] + f[-1])
        return gamma_value

    elif method is 'Weierstrass':   # for non-integer values
        euler_constant = 0.57721566490153286060
        t = np.arange(1, 20)
        gamma_value = (np.exp(-euler_constant * z) / z) * ((1 / (1 + z / t)) * np.exp(z / n)).prod()
        return gamma_value
    else:
        raise Exception('The method of calculating the Gamma function is not specified correctly!')


def Beta(x, y, method: str = 'numerical'):
    """
    calculates the Beta function B(a,y)
    :param x: a float value such that 0<x<1
    :param y: a float value such that 0<y<1
    :param method: the method used to calcualte the beta function
    :return:
    """
    def beta_kernel(x, y, t):
        return (t ** (x - 1)) * (1 - t)**(y-1)

    if method is 'numerical':
        t = np.linspace(0, 1, 10000)
        f = beta_kernel(x, y, t)
        deltat = t[1] - t[0]
        return deltat * (f[1:-1]).sum() + 0.5 * deltat * (f[0] + f[-1])


def Erf(z, method: str = 'numerical')->float:
    """
    The error function used to calculate the truncated gaussian distribution
    :param z: normalized input variable
    :return: the value of the error function
    """
    def erf_kernel(t):
        return (2/np.sqrt(np.pi)) * np.exp(-t**2)



    if method is 'fast':
        return (2 / (np.sqrt(np.pi))) * (z - (z ** 3 / 3) + (z ** 5 / 10) - (z ** 7 / 42) + (z ** 9 / 216))
    elif method is 'numerical':
        t = np.linspace(0, z, 10000)
        f = erf_kernel(t)
        deltat = t[1] - t[0]
        return deltat * (f[1:-1]).sum() + 0.5 * deltat * (f[0] + f[-1])
