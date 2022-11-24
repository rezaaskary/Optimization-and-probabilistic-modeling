from tensorflow_probability.substrates.jax import distributions
from jax import vmap, jit, grad, random, lax, scipy
import jax.numpy as jnp


class ContinuousDistributions:
    def __init__(self,
                 alpha: jnp.ndarray = None,
                 beta: jnp.ndarray = None,
                 lower: jnp.ndarray = None,
                 upper: jnp.ndarray = None,
                 loc: jnp.ndarray = None,
                 var: jnp.ndarray = None,
                 scale: jnp.ndarray = None,
                 variant_chains: bool = False,
                 activate_jit: bool = False,
                 n_chains: int = 1,
                 random_seed: int = 1,
                 fixed_parameters: bool = True) -> None:

        if isinstance(n_chains, int):
            self.n_chains = n_chains
        elif n_chains is None:
            self.n_chains = 1
        else:
            raise Exception(f'The value of upper is not specified correctly ({self.__class__} distribution)!')

        if isinstance(fixed_parameters, bool):
            self.fixed_parameters = fixed_parameters
        else:
            raise Exception(f'Please correctly specify the type of simulation (fixed or variant parameters in'
                            f' ({self.__class__} distribution)) !')

        if isinstance(random_seed, int):
            self.key = random.PRNGKey(random_seed)
        else:
            raise Exception(f'The random seed is not specified correctly ({self.__class__} distribution)!')

        if isinstance(variant_chains, bool):
            self.variant_chains = variant_chains
        else:
            raise Exception(f'Please specify whether the number of chains are fixed or variant during simulation '
                            f'({self.__class__} distribution)!')

        if isinstance(activate_jit, bool):
            self.activate_jit = activate_jit
        else:
            raise Exception(f'Please specify the activation of the just-in-time evaluation'
                            f' ({self.__class__} distribution)!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(lower, (jnp.ndarray, float, int)):
            self.lower = lower
        elif lower is None:
            self.lower = None
        else:
            raise Exception(f'The value of lower is not specified correctly ({self.__class__} distribution)!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(upper, (jnp.ndarray, float, int)):
            self.upper = upper
        elif upper is None:
            self.upper = None
        else:
            raise Exception(f'The value of lower is not specified correctly ({self.__class__} distribution)!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(loc, (jnp.ndarray, float, int)) and isinstance(var, (jnp.ndarray, float, int)):
            raise Exception(f'Please Enter either variance or standard deviation ({self.__class__} distribution)!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(scale, (jnp.ndarray, float, int)) and not isinstance(var, (jnp.ndarray, float, int)):
            if scale > 0:
                self.scale = scale
                self.var = scale ** 2
            else:
                raise Exception(f'The standard deviation should be a positive value ({self.__class__} distribution)!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if not isinstance(scale, (jnp.ndarray, float, int)) and isinstance(var, (jnp.ndarray, float, int)):
            if var > 0:
                self.scale = var ** 0.5
                self.variance = var
            else:
                raise Exception(f'The standard deviation should be a positive value ({self.__class__} distribution)!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if scale is None and var is None:
            self.sigma = None
            self.variance = None
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # if isinstance(nu, (jnp.ndarray, float, int)):
        #     self.nu = nu
        # elif nu is None:
        #     self.nu = None
        # else:
        #     raise Exception(f'The value of nu is not specified correctly ({self.__class__} distribution)!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(loc, (jnp.ndarray, float, int)):
            self.loc = loc
        elif loc is None:
            self.loc = None
        else:
            raise Exception(f'The value of loc is not specified correctly ({self.__class__} distribution)!')

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(alpha, (jnp.ndarray, float, int)):
            self.alpha = alpha
        elif alpha is None:
            self.alpha = None
        else:
            raise Exception(f'The value of alpha is not specified correctly ({self.__class__} distribution)!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(beta, (jnp.ndarray, float, int)):
            self.beta = beta
        elif beta is None:
            self.beta = None
        else:
            raise Exception(f'The value of beta is not specified correctly ({self.__class__} distribution)!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'mode': self.distance_function.mode(name='mode'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def parallelization(self):
        def probablity_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
            f"""
            The probability of variable x taken from {self.name} distribution
            :param x: An array with the size of (1xC)
            :return: The probability of the {self.name} distribution with the size of (1xC)
            """
            return self.distance_function.prob(value=x, name='prob')

        def cumulative_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
            f"""
            The cumulative function of {self.name} distribution
            :param x: An array with the size of (1xC) 
            :return: The cumulative distribution function of {self.name} distribution with the size of (1xC)
            """
            return self.distance_function.cdf(value=x, name='cdf')

        def log_probablity_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
            f"""
            The log probability of {self.name} distribution
            :param x: An array with the size of (1xC)
            :return: The log function of {self.name} distribution with the size of (1xC)
            """
            return self.distance_function.log_prob(value=x, name='log prob')

        def log_cumulative_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
            f"""
            The log of {self.name} distribution
            :param x: An array with the size of (1xC)
            :return: The log function of cumulative {self.name} distribution with the size of (1xC)
            """
            return self.distance_function.log_cdf(value=x, name='log cdf')

        def diff_probablity_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
            f"""
            Derivative of {self.name} distribution
            :param x: An array with the size of (1xC)
            :return: derivatives of {self.name} distribution with respect to variable x calculated in size of (1xC)
            """
            return (self.distance_function.prob(value=x, name='diff prob'))[0]

        def diff_cumulative_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
            f"""
            Derivative of cumulative function of {self.name} distribution
            :param x: An array with the size of (1xC)
            :return: derivatives of CDF of  {self.name} distribution with respect to variable x calculated in size
            of (1xC)
            """
            return (self.distance_function.cdf(value=x, name='diff cdf'))[0]

        def diff_log_probablity_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
            f"""
             Derivative of the  log of {self.name} distribution
            :param x: An array with the size of (1xC)
            :return: derivatives of the log of {self.name} distribution with respect to variable x calculated in
            size of (1xC)
            """
            return (self.distance_function.log_prob(value=x, name='diff log  prob'))[0]

        def diff_log_cumulative_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
            f"""
            Derivative of the  log of cumulative function of {self.name} distribution
            :param x: An array with the size of (1xC)
            :return: derivatives of the log CDF of {self.name} distribution with respect to variable x calculated in
            size of (1xC)
            """
            return (self.distance_function.log_cdf(value=x, name='diff log  cdf'))[0]

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        def mle(x: jnp.ndarray = None, checking_inputs: bool = False):
            """
            Enter an array of the data to fit the distribution of parameters using MLE
            :param x: Nx1 array of Data
            :param checking_inputs: A boolean variable used to activate checking the correctness of input variables
            :return: A dictionary of the distribution parameters
            """
            return self.distance_function.experimental_fit(value=x, validate_args=checking_inputs).parameters

        def sampling_from_distribution(sample_shape: tuple = None) -> jnp.ndarray:
            return self.distance_function.sample(sample_shape=sample_shape, seed=self.key, name='sample from pdf')

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        self.sample = sampling_from_distribution
        self.maximum_liklihood_estimation = mle
        if self.fixed_parameters:  # when the number of parallel evaluation is fixed. Useful for MCMC
            if self.activate_jit:  # activating jit
                self.pdf = jit(vmap(fun=probablity_distribution_, in_axes=self.vectorized_index_fcn, out_axes=1))
                self.diff_pdf = jit(vmap(grad(fun=diff_probablity_distribution_), in_axes=self.vectorized_index_diff_fcn
                                         , out_axes=0))
                self.log_pdf = jit(vmap(fun=log_probablity_distribution_, in_axes=self.vectorized_index_fcn,
                                        out_axes=1))
                self.diff_log_pdf = jit(vmap(grad(fun=diff_log_probablity_distribution_),
                                             in_axes=self.vectorized_index_diff_fcn, out_axes=0))
                self.cdf = jit(vmap(fun=cumulative_distribution_, in_axes=self.vectorized_index_fcn, out_axes=1))
                self.log_cdf = jit(vmap(fun=log_cumulative_distribution_, in_axes=self.vectorized_index_fcn,
                                        out_axes=1))
                self.diff_cdf = jit(vmap(grad(fun=diff_cumulative_distribution_), in_axes=self.vectorized_index_diff_fcn
                                         , out_axes=0))
                self.diff_log_cdf = jit(vmap(grad(fun=diff_log_cumulative_distribution_),
                                             in_axes=self.vectorized_index_diff_fcn, out_axes=0))
            else:  # Only using vectorized function
                self.pdf = vmap(fun=probablity_distribution_, in_axes=self.vectorized_index_fcn, out_axes=1)
                self.diff_pdf = vmap(grad(fun=diff_probablity_distribution_), in_axes=self.vectorized_index_diff_fcn
                                     , out_axes=0)
                self.log_pdf = vmap(fun=log_probablity_distribution_, in_axes=self.vectorized_index_fcn, out_axes=1)
                self.diff_log_pdf = vmap(grad(fun=diff_log_probablity_distribution_),
                                         in_axes=self.vectorized_index_diff_fcn, out_axes=0)
                self.cdf = vmap(fun=cumulative_distribution_, in_axes=self.vectorized_index_fcn, out_axes=1)
                self.log_cdf = vmap(fun=log_cumulative_distribution_, in_axes=self.vectorized_index_fcn, out_axes=1)
                self.diff_cdf = vmap(grad(fun=diff_cumulative_distribution_), in_axes=self.vectorized_index_diff_fcn
                                     , out_axes=0)
                self.diff_log_cdf = vmap(grad(fun=diff_log_cumulative_distribution_),
                                         in_axes=self.vectorized_index_diff_fcn, out_axes=0)


class Uniform(ContinuousDistributions):
    def __init__(self, lower: float = None, upper: float = None, activate_jit: bool = False,
                 random_seed: int = 1, fixed_parameters: bool = True, n_chains: int = 1) -> None:
        """
        In probability theory and statistics, the continuous uniform distribution or rectangular distribution is a
        family of symmetric probability distributions. The distribution describes an experiment where there is an
        arbitrary outcome that lies between certain bounds. The bounds are defined by the parameters, lower and upper,
        which are the minimum and maximum values.
        [1] Dekking, Michel (2005). A modern introduction to probability and statistics : understanding why and how.
        London, UK: Springer. pp. 60–61. ISBN 978-1-85233-896-1

        Continuous uniform distribution
        :param lower: The lower limit of uniform distribution
        :param upper: The upper limit of uniform distribution
        """
        # recalling parameter values from the main parent class
        super(Uniform, self).__init__(lower=lower, upper=upper, activate_jit=activate_jit, random_seed=random_seed,
                                      fixed_parameters=fixed_parameters, n_chains=n_chains)
        self.name = 'Uniform'
        # checking the correctness of the parameters
        if not isinstance(self.lower, type(self.upper)):
            raise Exception(f'The input parameters are not consistent ({self.name} distribution)!')
        if jnp.any(self.lower >= self.upper):
            raise Exception(f'The lower limit of the uniform distribution is greater than the upper limit'
                            f' ({self.name} distribution)!')

        if self.fixed_parameters:  # specifying the main probability function for invariant simulation
            self.distance_function = distributions.Uniform(low=self.lower, high=self.upper, name='Uniform')
            self.vectorized_index_fcn = [1]  # input x, parameter 1, parameter 2
            self.vectorized_index_diff_fcn = [0]
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'mode': self.distance_function.mode(name='mode'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Normal(ContinuousDistributions):
    def __init__(self, scale: float = None, var: float = None, loc: float = None,
                 random_seed: int = 1, activate_jit: bool = False, fixed_parameters: bool = True,
                 n_chains: int = 1) -> None:

        self.name = 'Normal'

        # recalling parameter values from the main parent class
        super(Normal, self).__init__(scale=scale, var=var, loc=loc,
                                     activate_jit=activate_jit, random_seed=random_seed,
                                     fixed_parameters=fixed_parameters, n_chains=n_chains)

        # checking the correctness of the parameters
        if self.loc is None or self.scale is None:
            raise Exception(f'The value of either mean or standard deviation is not specified'
                            f'({self.name} distribution)!')

        if self.fixed_parameters:  # specifying the main probability function for invariant simulation
            self.distance_function = distributions.Normal(loc=self.loc, scale=self.scale, name='Normal')
            self.vectorized_index_fcn = [1]  # input x, parameter 1, parameter 2
            self.vectorized_index_diff_fcn = [0]
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'mode': self.distance_function.mode(name='mode'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class TruncatedNormal(ContinuousDistributions):
    def __init__(self, lower: float = None, upper: float = None, loc: float = None, var: float = None,
                 scale: float = None, activate_jit: bool = False, random_seed: int = 1,
                 fixed_parameters: bool = True, n_chains: int = 1) -> None:

        self.name = 'Truncated Normal'

        # recalling parameter values from the main parent class
        super(TruncatedNormal, self).__init__(loc=loc, var=var, scale=scale, lower=lower, upper=upper
                                              , activate_jit=activate_jit, random_seed=random_seed,
                                              fixed_parameters=fixed_parameters, n_chains=n_chains)

        # checking the correctness of the parameters
        if self.loc is None or self.scale is None:
            raise Exception(f'The value of either mean or standard deviation is not specified'
                            f' ({self.name} distribution)!')

        if self.lower >= self.upper:
            raise Exception(f'The lower bound of the distribution cannot be greater than the upper bound'
                            f' ({self.name} distribution)!')

        if self.fixed_parameters:  # specifying the main probability function for invariant simulation
            self.distance_function = distributions.TruncatedNormal(loc=self.loc, scale=self.scale,
                                                                   low=self.lower, high=self.upper, name=self.name)
            self.vectorized_index_fcn = [1]  # input x, parameter 1, parameter 2
            self.vectorized_index_diff_fcn = [0]
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'mode': self.distance_function.mode(name='mode'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class HalfNormal(ContinuousDistributions):
    def __init__(self, scale: float = None, var: float = None,
                 activate_jit: bool = False, random_seed: int = 1, fixed_parameters: bool = True,
                 n_chains: int = 1) -> None:

        self.name = 'Half Normal'

        # recalling parameter values from the main parent class
        super(HalfNormal, self).__init__(scale=scale, var=var
                                         , activate_jit=activate_jit, random_seed=random_seed,
                                         fixed_parameters=fixed_parameters, n_chains=n_chains)

        # checking the correctness of the parameters
        if self.var is None or self.scale is None:
            raise Exception(
                f'The value of either variance or standard deviation is not specified ({self.name} distribution)!')

        if self.fixed_parameters:  # specifying the main probability function for invariant simulation
            self.distance_function = distributions.HalfNormal(scale=self.scale, name=self.name)
            self.vectorized_index_fcn = [1]  # input x, parameter 1, parameter 2
            self.vectorized_index_diff_fcn = [0]
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'mode': self.distance_function.mode(name='mode'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class TwoPieceNormal(ContinuousDistributions):
    def __init__(self, scale: float = None, loc: float = None, var: float = None, alpha: float = None,
                 activate_jit: bool = False, random_seed: int = 1, fixed_parameters: bool = True,
                 n_chains: int = 1) -> None:

        self.name = 'TwoPieceNormal'

        # recalling parameter values from the main parent class
        super(TwoPieceNormal, self).__init__(var=var, loc=loc, scale=scale, alpha=alpha, n_chains=n_chains
                                             , activate_jit=activate_jit, random_seed=random_seed,
                                             fixed_parameters=fixed_parameters)

        # checking the correctness of the parameters
        if self.alpha < 0:
            raise Exception(f'The input parameters alpha is not sacrificed correctly ({self.name} Distribution)!')

        if self.fixed_parameters:  # specifying the main probability function for invariant simulation
            self.distance_function = distributions.TwoPieceNormal(scale=self.scale, loc=self.loc, skewness=self.alpha,
                                                                  name=self.name)
            self.vectorized_index_fcn = [1]  # input x, parameter 1, parameter 2
            self.vectorized_index_diff_fcn = [0]
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Beta(ContinuousDistributions):
    def __init__(self, alpha: float = None, beta: float = None,
                 activate_jit: bool = False, random_seed: int = 1, fixed_parameters: bool = True,
                 n_chains: int = 1) -> None:

        self.name = 'Beta'

        # recalling parameter values from the main parent class
        super(Beta, self).__init__(alpha=alpha, beta=beta
                                   , activate_jit=activate_jit, random_seed=random_seed,
                                   fixed_parameters=fixed_parameters, n_chains=n_chains)

        # checking the correctness of the parameters
        if self.alpha < 0:
            raise Exception(f'The input parameters alpha is not sacrificed correctly ({self.name} Distribution)!')
        if self.beta < 0:
            raise Exception(f'The input parameters beta is not sacrificed correctly ({self.name} Distribution)!')
        if self.fixed_parameters:  # specifying the main probability function for invariant simulation
            self.distance_function = distributions.Beta(force_probs_to_zero_outside_support=True,
                                                        concentration0=self.beta, concentration1=self.alpha,
                                                        name=self.name)
            self.vectorized_index_fcn = [1]  # input x, parameter 1, parameter 2
            self.vectorized_index_diff_fcn = [0]
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'mode': self.distance_function.mode(name='mode'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Kumaraswamy(ContinuousDistributions):
    def __init__(self, alpha: float = None, beta: float = None,
                 activate_jit: bool = False, random_seed: int = 1, fixed_parameters: bool = True,
                 n_chains: int = 1) -> None:

        self.name = 'Kumaraswamy'

        # recalling parameter values from the main parent class
        super(Kumaraswamy, self).__init__(alpha=alpha, beta=beta
                                          , activate_jit=activate_jit, random_seed=random_seed,
                                          fixed_parameters=fixed_parameters, n_chains=n_chains)

        # checking the correctness of the parameters
        if self.alpha <= 1:
            raise Exception(f'The input parameters alpha is not sacrificed correctly ({self.name} Distribution)!')
        if self.beta <= 1:
            raise Exception(f'The input parameters beta is not sacrificed correctly ({self.name} Distribution)!')

        if self.fixed_parameters:  # specifying the main probability function for invariant simulation
            self.distance_function = distributions.Kumaraswamy(concentration1=self.alpha,
                                                               concentration0=self.beta,
                                                               name=self.name)
            self.vectorized_index_fcn = [1]  # input x, parameter 1, parameter 2
            self.vectorized_index_diff_fcn = [0]
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'mode': self.distance_function.mode(name='mode'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information


##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# KK = Uniform(lower=2, upper=4, activate_jit=True, random_seed=6, fixed_parameters=True)
# KK = Normal(scale=2, loc=3, activate_jit=True)
# KK = TruncatedNormal(scale=2, loc=3, lower=-2, upper=4, activate_jit=True)
# KK = HalfNormal(scale=2, activate_jit=True)
# KK = TwoPieceNormal(scale=3, loc=1, activate_jit=False, alpha=2)
KK = Beta(alpha=2, beta=3, activate_jit=False)
x = random.uniform(key=random.PRNGKey(7), minval=-20, maxval=20, shape=(1000, 1), dtype=jnp.float64)
# TT = E.pdf(x)
# TT2 = E.log(x)

E1 = KK.pdf(x)
E6 = KK.diff_pdf(x)
E2 = KK.log_pdf(x)

E3 = KK.diff_log_pdf(x)
E4 = KK.cdf(x)
E5 = KK.log_cdf(x)
E8 = KK.diff_cdf(x)
E9 = KK.diff_log_cdf(x)
E11 = KK.statistics
E12 = KK.sample(sample_shape=(2000, 1))
E10 = KK.maximum_liklihood_estimation(x=x)

E10

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# class PDF(ContinuousDistributions):
#     def __init__(self, : float = None, : float = None,
#                  activate_jit: bool = False, random_seed: int = 1, fixed_parameters: bool = True,
#                  n_chains: int = 1) -> None:
#
#         self.name = 'U'
#
#         # recalling parameter values from the main parent class
#         super(PDF, self).__init__(
#             , activate_jit=activate_jit, random_seed=random_seed,
#             fixed_parameters=fixed_parameters, n_chains=n_chains)
#
#
#         # checking the correctness of the parameters
#         if self.alpha < 0:
#             raise Exception(f'The input parameters alpha is not sacrificed correctly ({self.name} Distribution)!')
#
#         if self.fixed_parameters:  # specifying the main probability function for invariant simulation
#             self.distance_function = distributions.(, name=self.name)
#             self.vectorized_index_fcn = [1]  # input x, parameter 1, parameter 2
#             self.vectorized_index_diff_fcn = [0]
#         ContinuousDistributions.parallelization(self)
#     @property
#     def statistics(self):
#         information = {'mean': self.distance_function.mean(name='mean'),
#                        'mode': self.distance_function.mode(name='mode'),
#                        'entropy': self.distance_function.entropy(name='entropy'),
#                        'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
#                        'median': self.distance_function.quantile(value=0.5, name='median'),
#                        'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
#                        'range': self.distance_function.range(name='range'),
#                        'std': self.distance_function.stddev(name='stddev'),
#                        'var': self.distance_function.variance(name='variance'),
#                        }
#         return information
