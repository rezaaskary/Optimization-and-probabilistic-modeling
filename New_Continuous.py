from tensorflow_probability.substrates.jax import distributions
from jax import vmap, jit, grad, random, lax, scipy
import jax.numpy as jnp


class ContinuousDistributions:
    def __init__(self,
                 lower: jnp.ndarray = None,
                 upper: jnp.ndarray = None,
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
            raise Exception('The value of upper is not specified correctly!')

        if isinstance(fixed_parameters, bool):
            self.fixed_parameters = fixed_parameters
        else:
            raise Exception('Please correctly specify the type of simulation (fixed  or variant parameters ) !')

        if isinstance(random_seed, int):
            self.key = random.PRNGKey(random_seed)
        else:
            raise Exception('The random seed is not specified correctly!')

        if isinstance(variant_chains, bool):
            self.variant_chains = variant_chains
        else:
            raise Exception('Please specify whether the number of chains are fixed or variant during simulation !')

        if isinstance(activate_jit, bool):
            self.activate_jit = activate_jit
        else:
            raise Exception('Please specify the activation of the just-in-time evaluation!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(lower, (jnp.ndarray, float, int)):
            self.lower = lower
        elif lower is None:
            self.lower = None
        else:
            raise Exception('The value of lower is not specified correctly!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(upper, (jnp.ndarray, float, int)):
            self.upper = upper
        elif upper is None:
            self.upper = None
        else:
            raise Exception('The value of lower is not specified correctly!')
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
                self.pdf = jit(vmap(fun=probablity_distribution_, in_axes=self.vectorized_index, out_axes=1))
                self.diff_pdf = jit(vmap(grad(fun=diff_probablity_distribution_), in_axes=[0], out_axes=0))
                self.log_pdf = jit(vmap(fun=log_probablity_distribution_, in_axes=self.vectorized_index, out_axes=1))
                self.diff_log_pdf = jit(vmap(grad(fun=diff_log_probablity_distribution_), in_axes=[0], out_axes=0))
                self.cdf = jit(vmap(fun=cumulative_distribution_, in_axes=self.vectorized_index, out_axes=1))
                self.log_cdf = jit(vmap(fun=log_cumulative_distribution_, in_axes=self.vectorized_index, out_axes=1))
                self.diff_cdf = jit(vmap(grad(fun=diff_cumulative_distribution_), in_axes=[0], out_axes=0))
                self.diff_log_cdf = jit(vmap(grad(fun=diff_log_cumulative_distribution_), in_axes=[0], out_axes=0))
            else:  # Only using vectorized function
                self.pdf = vmap(fun=probablity_distribution_, in_axes=self.vectorized_index, out_axes=1)
                self.diff_pdf = vmap(grad(fun=diff_probablity_distribution_), in_axes=[0], out_axes=0)
                self.log_pdf = vmap(fun=log_probablity_distribution_, in_axes=self.vectorized_index, out_axes=1)
                self.diff_log_pdf = vmap(grad(fun=diff_log_probablity_distribution_), in_axes=[0], out_axes=0)
                self.cdf = vmap(fun=cumulative_distribution_, in_axes=self.vectorized_index, out_axes=1)
                self.log_cdf = vmap(fun=log_cumulative_distribution_, in_axes=self.vectorized_index, out_axes=1)
                self.diff_cdf = vmap(grad(fun=diff_cumulative_distribution_), in_axes=[0], out_axes=0)
                self.diff_log_cdf = vmap(grad(fun=diff_log_cumulative_distribution_), in_axes=[0], out_axes=0)


class Uniform(ContinuousDistributions):
    def __init__(self, lower: float = None, upper: float = None, activate_jit: bool = False,
                 random_seed: int = 1, fixed_parameters: bool = True) -> None:
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
                                      fixed_parameters=fixed_parameters)
        self.name = 'Uniform'
        # checking the correctness of the parameters
        if not isinstance(self.lower, type(self.upper)):
            raise Exception('The input parameters are not consistent (Uniform Distribution)!')
        if jnp.any(self.lower >= self.upper):
            raise Exception('The lower limit of the uniform distribution is greater than the upper limit!')

        if self.fixed_parameters:  # specifying the main probability function for invariant simulation
            self.distance_function = distributions.Uniform(low=self.lower, high=self.upper, name='Uniform')
            self.vectorized_index = [1]  # input x, parameter 1, parameter 2

        ContinuousDistributions.parallelization(self)
        # else:  # specifying the main function for the variant simulation
        #     self.variant_parameters = jnp.concatenate(arrays=(self.lower, self.upper), axis=1)
        #
        #     def variant_function(distribution_parameters: jnp.ndarray = self.variant_parameters):
        #         return distributions.Uniform(low=distribution_parameters[:, 0], high=distribution_parameters[:, 1],
        #                                      name='Uniform')
        #
        #     self.distance_function = variant_function
        #     self.vectorized_index = jnp.array([1, 1, 1], dtype=int)  # input x, parameter 1, parameter 2


        # x = random.uniform(key=random.PRNGKey(7), minval=0.01, maxval=20, shape=(1000, 1), dtype=jnp.float64)
        # x = x.at[5,0].set(jnp.nan)
        # PP = self.distance_function.experimental_fit(value=x, validate_args=True)
        # PP = (self.distance_function.sample(sample_shape=(100,), seed=self.key))

        # def ddz(x, ub, lb):
        #     return distributions.Uniform(low=lb, high=ub, name='Uniform').prob(x)
        #
        # PP2 = vmap(ddz, in_axes=[0, 0, 0], out_axes=0)(x, x + 4, x - 3)










KK = Uniform(lower=2, upper=4, activate_jit=True, random_seed=6, fixed_parameters=True)
x = random.uniform(key=random.PRNGKey(7), minval=0.01, maxval=20, shape=(1000, 1), dtype=jnp.float64)
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
E10 = KK.maximum_liklihood_estimation(x=x)
E10