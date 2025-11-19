"""
Optimization and Probabilistic Modeling Package
================================================

A comprehensive CPU/GPU accelerated Python package for probabilistic programming, 
Bayesian inference, and optimization.

Modules
-------
- optimization_function : Gradient-based optimizers (SGD, Adam, RMSprop) and convex optimization
- Models : Statistical models (PCA, PPCA, Factor Analysis, CCA)
- sampler_algorithms : MCMC sampling (Metropolis-Hastings, MCMC Hammer)
- Continuous : Continuous probability distributions (JAX-accelerated)
- Discrete : Discrete probability distributions (JAX-accelerated)
- Probablity_distributions : Legacy probability distributions (NumPy-based)
- Ordinary_Differential_Equation_Solvers : ODE solvers for dynamical systems
- Liklihoods : Likelihood functions for statistical inference
- Feature_Selection : Feature selection methods
- Sensitivity_Analysis : Sensitivity analysis tools
- mathmatics : Mathematical utility functions

Author
------
Mohammad Reza Askari
maskari@hawk.iit.edu

License
-------
MIT License
"""

__version__ = '0.1.0'
__author__ = 'Mohammad Reza Askari'
__email__ = 'maskari@hawk.iit.edu'

# Import key classes and functions for easy access
from .optimization_function import Optimizer, Convex_problems_dual_ascend, ADMM
from .Models import PPCA, PCA, FactorAnalysis, CanonicalCorrelationAnalysis
from .sampler_algorithms import (
    MetropolisHastings, 
    MCMCHammer, 
    ModelParallelizer,
    ParameterProposalInitialization
)
from .Ordinary_Differential_Equation_Solvers import ODESolvers

# JAX-accelerated distributions
try:
    from .Continuous import (
        Uniform, Normal, TruncatedNormal, HalfNormal,
        Beta, Kumaraswamy, Exponential, Laplace,
        StudentT, HalfStudentT, Cauchy, HalfCauchy,
        Gamma, InverseGamma, Weibull, Chi2,
        LogNormal, Pareto, ExponentiallyModifiedGaussian,
        Triangular, Gumbel, Logistic, LogitNormal,
        TruncatedCauchy, VonMises, Bates, TwoPieceNormal
    )
    from .Discrete import Binomial, BetaBinomial, Bernoulli, Poisson
except ImportError:
    print("Warning: JAX-accelerated distributions not available. Install JAX and TensorFlow Probability.")

__all__ = [
    # Optimization
    'Optimizer',
    'Convex_problems_dual_ascend',
    'ADMM',
    # Statistical Models
    'PPCA',
    'PCA',
    'FactorAnalysis',
    'CanonicalCorrelationAnalysis',
    # MCMC Sampling
    'MetropolisHastings',
    'MCMCHammer',
    'ModelParallelizer',
    'ParameterProposalInitialization',
    # ODE Solvers
    'ODESolvers',
    # Continuous Distributions
    'Uniform',
    'Normal',
    'TruncatedNormal',
    'HalfNormal',
    'Beta',
    'Kumaraswamy',
    'Exponential',
    'Laplace',
    'StudentT',
    'HalfStudentT',
    'Cauchy',
    'HalfCauchy',
    'Gamma',
    'InverseGamma',
    'Weibull',
    'Chi2',
    'LogNormal',
    'Pareto',
    'ExponentiallyModifiedGaussian',
    'Triangular',
    'Gumbel',
    'Logistic',
    'LogitNormal',
    'TruncatedCauchy',
    'VonMises',
    'Bates',
    'TwoPieceNormal',
    # Discrete Distributions
    'Binomial',
    'BetaBinomial',
    'Bernoulli',
    'Poisson',
]
