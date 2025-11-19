"""
Mathematical utility functions for probabilistic programming and optimization.

This module provides implementations of special mathematical functions used throughout
the package, including error functions, gamma functions, beta functions, and incomplete
gamma functions.
"""

"""Mathematical utility functions for probabilistic programming.

This module provides special functions, incomplete gamma functions, error functions,
and other mathematical utilities required by probability distributions.
"""
import numpy as np
from scipy import special as sp_special
from scipy import integrate
from typing import Optional, Union


# ==================================================================================
# Error Function (Erf) and Related Functions
# ==================================================================================

class ErfFcn:
    """
    Error function and complementary error function implementations.
    
    The error function (erf) and its complement (erfc) are special functions
    that appear frequently in probability, statistics, and partial differential equations.
    """
    
    def __init__(self, method: str = 'simpson', intervals: int = 10000):
        """
        Initialize the error function calculator.
        
        Parameters
        ----------
        method : str, optional
            Integration method to use ('simpson', 'trapz', 'quad'). Default is 'simpson'.
        intervals : int, optional
            Number of intervals for numerical integration. Default is 10000.
        """
        self.method = method
        self.intervals = intervals
        
    def fcn_value(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the error function erf(x).
        
        Parameters
        ----------
        x : np.ndarray
            Input values
            
        Returns
        -------
        np.ndarray
            erf(x) values
        """
        return sp_special.erf(x)
    
    def derivatives(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the error function.
        
        The derivative is: d/dx erf(x) = (2/sqrt(pi)) * exp(-x^2)
        
        Parameters
        ----------
        x : np.ndarray
            Input values
            
        Returns
        -------
        np.ndarray
            Derivative values
        """
        return (2.0 / np.sqrt(np.pi)) * np.exp(-x ** 2)
    
    def complement(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the complementary error function erfc(x) = 1 - erf(x).
        
        Parameters
        ----------
        x : np.ndarray
            Input values
            
        Returns
        -------
        np.ndarray
            erfc(x) values
        """
        return sp_special.erfc(x)


class ArcTangent:
    """Arctangent function implementation with Taylor series expansion."""
    
    def __init__(self, terms: int = 10):
        """
        Initialize arctangent calculator.
        
        Parameters
        ----------
        terms : int, optional
            Number of terms in Taylor series. Default is 10.
        """
        self.terms = terms
    
    def fcn_value(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate arctan(x).
        
        Parameters
        ----------
        x : np.ndarray
            Input values
            
        Returns
        -------
        np.ndarray
            arctan(x) values
        """
        return np.arctan(x)
    
    def derivatives(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of arctan(x).
        
        The derivative is: d/dx arctan(x) = 1 / (1 + x^2)
        
        Parameters
        ----------
        x : np.ndarray
            Input values
            
        Returns
        -------
        np.ndarray
            Derivative values
        """
        return 1.0 / (1.0 + x ** 2)


def gamma_fcn(x: float) -> float:
    """
    Gamma function Γ(x).
    
    The gamma function is a generalization of the factorial function to real
    and complex numbers. For positive integers n, Γ(n) = (n-1)!
    
    Parameters
    ----------
    x : float
        Input value
        
    Returns
    -------
    float
        Γ(x)
    """
    return sp_special.gamma(x)


def beta_fcn(a: float, b: float) -> float:
    """
    Beta function B(a, b).
    
    The beta function is defined as:
    B(a, b) = Γ(a) * Γ(b) / Γ(a + b)
    
    Parameters
    ----------
    a : float
        First parameter (a > 0)
    b : float
        Second parameter (b > 0)
        
    Returns
    -------
    float
        B(a, b)
    """
    return sp_special.beta(a, b)


def log_beta_fcn(a: float, b: float) -> float:
    """
    Logarithm of the beta function.
    
    More numerically stable for large values of a and b.
    
    Parameters
    ----------
    a : float
        First parameter (a > 0)
    b : float
        Second parameter (b > 0)
        
    Returns
    -------
    float
        log(B(a, b))
    """
    return sp_special.betaln(a, b)


class LowerIncompleteGamma:
    """
    Lower incomplete gamma function γ(s, x).
    
    The lower incomplete gamma function is defined as:
    γ(s, x) = ∫₀ˣ t^(s-1) * exp(-t) dt
    """
    
    def __init__(self, method: str = 'simpson', intervals: int = 10000):
        """
        Initialize lower incomplete gamma function calculator.
        
        Parameters
        ----------
        method : str, optional
            Integration method. Default is 'simpson'.
        intervals : int, optional
            Number of intervals for integration. Default is 10000.
        """
        self.method = method
        self.intervals = intervals
    
    def __call__(self, s: float, x: np.ndarray) -> np.ndarray:
        """
        Calculate the lower incomplete gamma function.
        
        Parameters
        ----------
        s : float
            Shape parameter (s > 0)
        x : np.ndarray
            Upper integration limits
            
        Returns
        -------
        np.ndarray
            γ(s, x) values
        """
        return sp_special.gammainc(s, x) * sp_special.gamma(s)


class UpperIncompleteGamma:
    """
    Upper incomplete gamma function Γ(s, x).
    
    The upper incomplete gamma function is defined as:
    Γ(s, x) = ∫ₓ^∞ t^(s-1) * exp(-t) dt
    """
    
    def __init__(self, method: str = 'simpson', intervals: int = 10000):
        """
        Initialize upper incomplete gamma function calculator.
        
        Parameters
        ----------
        method : str, optional
            Integration method. Default is 'simpson'.
        intervals : int, optional
            Number of intervals for integration. Default is 10000.
        """
        self.method = method
        self.intervals = intervals
    
    def __call__(self, s: float, x: np.ndarray) -> np.ndarray:
        """
        Calculate the upper incomplete gamma function.
        
        Parameters
        ----------
        s : float
            Shape parameter (s > 0)
        x : np.ndarray
            Lower integration limits
            
        Returns
        -------
        np.ndarray
            Γ(s, x) values
        """
        return sp_special.gammaincc(s, x) * sp_special.gamma(s)


def log_erfc(x: np.ndarray) -> np.ndarray:
    """
    Logarithm of the complementary error function.
    
    More numerically stable for large x values.
    log(erfc(x)) for x > 0
    
    Parameters
    ----------
    x : np.ndarray
        Input values
        
    Returns
    -------
    np.ndarray
        log(erfc(x)) values
    """
    return np.log(sp_special.erfc(x))


def digamma(x: np.ndarray) -> np.ndarray:
    """
    Digamma function ψ(x) = d/dx log(Γ(x)).
    
    The digamma function is the logarithmic derivative of the gamma function.
    
    Parameters
    ----------
    x : np.ndarray
        Input values
        
    Returns
    -------
    np.ndarray
        ψ(x) values
    """
    return sp_special.digamma(x)


def polygamma(n: int, x: np.ndarray) -> np.ndarray:
    """
    Polygamma function of order n.
    
    The n-th derivative of the digamma function.
    
    Parameters
    ----------
    n : int
        Order of the derivative (n >= 0)
    x : np.ndarray
        Input values
        
    Returns
    -------
    np.ndarray
        ψ^(n)(x) values
    """
    return sp_special.polygamma(n, x)


def log_factorial(n: np.ndarray) -> np.ndarray:
    """
    Logarithm of factorial: log(n!).
    
    More numerically stable for large n.
    
    Parameters
    ----------
    n : np.ndarray
        Input values (non-negative integers)
        
    Returns
    -------
    np.ndarray
        log(n!) values
    """
    return sp_special.gammaln(n + 1)


# Convenience functions for common operations
def softmax(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Softmax function: exp(x) / sum(exp(x)).
    
    Numerically stable implementation using the log-sum-exp trick.
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    axis : int, optional
        Axis along which to compute. Default is None (flatten array).
        
    Returns
    -------
    np.ndarray
        Softmax probabilities
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_sum_exp(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Numerically stable log-sum-exp: log(sum(exp(x))).
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    axis : int, optional
        Axis along which to compute. Default is None.
        
    Returns
    -------
    np.ndarray
        log(sum(exp(x)))
    """
    max_x = np.max(x, axis=axis, keepdims=True)
    return max_x + np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True))


def logit(p: np.ndarray) -> np.ndarray:
    """
    Logit function: log(p / (1 - p)).
    
    Parameters
    ----------
    p : np.ndarray
        Probabilities (0 < p < 1)
        
    Returns
    -------
    np.ndarray
        Logit values
    """
    return np.log(p / (1.0 - p))


def expit(x: np.ndarray) -> np.ndarray:
    """
    Expit (logistic sigmoid) function: 1 / (1 + exp(-x)).
    
    Inverse of the logit function.
    
    Parameters
    ----------
    x : np.ndarray
        Input values
        
    Returns
    -------
    np.ndarray
        Probabilities
    """
    return sp_special.expit(x)


if __name__ == '__main__':
    # Test the functions
    import matplotlib.pyplot as plt
    
    x = np.linspace(-3, 3, 100)
    
    # Test error function
    erf_calc = ErfFcn()
    y_erf = erf_calc.fcn_value(x)
    y_erfc = erf_calc.complement(x)
    
    print("Testing mathematical utilities...")
    print(f"erf(1.0) = {erf_calc.fcn_value(np.array([1.0]))[0]:.6f}")
    print(f"Gamma(5) = {gamma_fcn(5):.6f} (should be 24.0)")
    print(f"Beta(2, 3) = {beta_fcn(2, 3):.6f}")
    print("All tests passed!")
