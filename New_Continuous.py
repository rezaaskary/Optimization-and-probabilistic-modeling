from tensorflow_probability.substrates.jax import distributions
from jax import vmap, jit, grad, random, lax, scipy, jacfwd
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
                 rate: jnp.ndarray = None,
                 df: jnp.ndarray = None,
                 kappa: jnp.ndarray = None,
                 lambd: jnp.ndarray = None,
                 peak: jnp.ndarray = None,
                 variant_chains: bool = False,
                 activate_jit: bool = False,
                 n_chains: int = 1,
                 random_seed: int = 1,
                 in_vec_dim: int = 1,
                 validate_input_range: bool = True,
                 out_vec_dim: int = 1) -> None:

        if isinstance(validate_input_range, bool):
            self.validate_input_range = validate_input_range
        else:
            raise Exception(f'Please specify whether the input should be validated before evaluation '
                            f'({self.__class__} distribution)!')

        if isinstance(in_vec_dim, int):
            self.in_vec_dim = in_vec_dim
        else:
            raise Exception(f'The value of upper is not specified correctly ({self.__class__} distribution)!')

        if isinstance(out_vec_dim, int):
            self.out_vec_dim = out_vec_dim
        else:
            raise Exception(f'The value of upper is not specified correctly ({self.__class__} distribution)!')

        if isinstance(n_chains, int):
            self.n_chains = n_chains
        elif n_chains is None:
            self.n_chains = 1
        else:
            raise Exception(f'The value of upper is not specified correctly ({self.__class__} distribution)!')

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
        if isinstance(lower, jnp.ndarray):
            self.lower = lower
        elif lower is None:
            self.lower = None
        else:
            raise Exception(f'The value of input parameters is not specified correctly. Please enter parameters  in the'
                            f' format of ndarrauy ({self.__class__} distribution)!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(kappa, jnp.ndarray):
            self.kappa = kappa
        elif kappa is None:
            self.kappa = None
        else:
            raise Exception(f'The value of input parameters is not specified correctly. Please enter parameters  in the'
                            f' format of ndarrauy ({self.__class__} distribution)!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(lambd, jnp.ndarray):
            self.lambd = lambd
        elif kappa is None:
            self.lambd = None
        else:
            raise Exception(f'The value of input parameters is not specified correctly. Please enter parameters  in the'
                            f' format of ndarrauy ({self.__class__} distribution)!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(peak, jnp.ndarray):
            self.peak = peak
        elif peak is None:
            self.peak = None
        else:
            raise Exception(f'The value of input parameters is not specified correctly. Please enter parameters  in the'
                            f' format of ndarrauy ({self.__class__} distribution)!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(upper, jnp.ndarray):
            self.upper = upper
        elif upper is None:
            self.upper = None
        else:
            raise Exception(f'The value of input parameters is not specified correctly. Please enter parameters  in the'
                            f' format of ndarrauy ({self.__class__} distribution)!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(df, jnp.ndarray):
            self.df = df
        elif df is None:
            self.df = None
        else:
            raise Exception(f'The value of input parameters is not specified correctly. Please enter parameters  in the'
                            f' format of ndarrauy ({self.__class__} distribution)!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(rate, jnp.ndarray):
            self.rate = rate
        elif rate is None:
            self.rate = None
        else:
            raise Exception(f'The value of input parameters is not specified correctly. Please enter parameters  in the'
                            f' format of ndarrauy ({self.__class__} distribution)!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(loc, jnp.ndarray) and isinstance(var, jnp.ndarray):
            raise Exception(f'Please Enter either variance or standard deviation ({self.__class__} distribution)!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(scale, jnp.ndarray) and not isinstance(var, jnp.ndarray):
            if jnp.any(scale > 0):
                self.scale = scale
                self.var = scale ** 2
            else:
                raise Exception(f'The standard deviation should be a positive value ({self.__class__} distribution)!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if not isinstance(scale, jnp.ndarray) and isinstance(var, jnp.ndarray):
            if jnp.arra(var > 0):
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
        if isinstance(loc, jnp.ndarray):
            self.loc = loc
        elif loc is None:
            self.loc = None
        else:
            raise Exception(f'The value of input parameters is not specified correctly. Please enter parameters  in the'
                            f' format of ndarrauy ({self.__class__} distribution)!')

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(alpha, jnp.ndarray):
            self.alpha = alpha
        elif alpha is None:
            self.alpha = None
        else:
            raise Exception(f'The value of input parameters is not specified correctly. Please enter parameters  in the'
                            f' format of ndarrauy ({self.__class__} distribution)!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(beta, jnp.ndarray):
            self.beta = beta
        elif beta is None:
            self.beta = None
        else:
            raise Exception(f'The value of input parameters is not specified correctly. Please enter parameters  in the'
                            f' format of ndarrauy ({self.__class__} distribution)!')
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

        if self.validate_input_range:
            def probability_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
                f"""
                The probability of variable x taken from the distribution
                :param x: An array with the size of (1xC)
                :return: The probability of the distribution with the size of (1xC)
                """
                return self.distance_function.prob(value=self.valid_range(x), name='prob')

            def cumulative_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
                f"""
                The cumulative function of the distribution
                :param x: An array with the size of (1xC) 
                :return: The cumulative distribution function of the distribution with the size of (1xC)
                """
                return self.distance_function.cdf(value=self.valid_range(x), name='cdf')

            def log_probability_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
                f"""
                The log probability of the distribution
                :param x: An array with the size of (1xC)
                :return: The log function of the distribution with the size of (1xC)
                """
                return self.distance_function.log_prob(value=self.valid_range(x), name='log prob')

            def log_cumulative_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
                f"""
                The log of the distribution
                :param x: An array with the size of (1xC)
                :return: The log function of cumulative the distribution with the size of (1xC)
                """
                return self.distance_function.log_cdf(value=self.valid_range(x), name='log cdf')

            def diff_probability_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
                f"""
                Derivative of the distribution
                :param x: An array with the size of (1xC)
                :return: derivatives of the distribution with respect to variable x calculated in size of (1xC)
                """
                return self.distance_function.prob(value=self.valid_range(x), name='diff prob')

            def diff_cumulative_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
                f"""
                Derivative of cumulative function of the distribution
                :param x: An array with the size of (1xC)
                :return: derivatives of CDF of the distribution with respect to variable x calculated in size
                of (1xC)
                """
                return self.distance_function.cdf(value=self.valid_range(x), name='diff cdf')

            def diff_log_probability_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
                f"""
                 Derivative of the  log of distribution
                :param x: An array with the size of (1xC)
                :return: derivatives of the log of the distribution with respect to variable x calculated in
                size of (1xC)
                """
                return self.distance_function.log_prob(value=self.valid_range(x), name='diff log  prob')

            def diff_log_cumulative_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
                f"""
                Derivative of the log of cumulative function of the distribution
                :param x: An array with the size of (1xC)
                :return: derivatives of the log CDF of the distribution with respect to variable x calculated in
                size of (1xC)
                """
                return self.distance_function.log_cdf(value=self.valid_range(x), name='diff log  cdf')
        else:

            def probability_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
                f"""
                The probability of variable x taken from the distribution
                :param x: An array with the size of (1xC)
                :return: The probability of the distribution with the size of (1xC)
                """
                return self.distance_function.prob(value=x, name='prob')

            def cumulative_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
                f"""
                The cumulative function of the distribution
                :param x: An array with the size of (1xC) 
                :return: The cumulative distribution function of the distribution with the size of (1xC)
                """
                return self.distance_function.cdf(value=x, name='cdf')

            def log_probability_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
                f"""
                The log probability of the distribution
                :param x: An array with the size of (1xC)
                :return: The log function of the distribution with the size of (1xC)
                """
                return self.distance_function.log_prob(value=x, name='log prob')

            def log_cumulative_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
                f"""
                The log of the distribution
                :param x: An array with the size of (1xC)
                :return: The log function of cumulative the distribution with the size of (1xC)
                """
                return self.distance_function.log_cdf(value=x, name='log cdf')

            def diff_probability_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
                f"""
                Derivative of the distribution
                :param x: An array with the size of (1xC)
                :return: derivatives of the distribution with respect to variable x calculated in size of (1xC)
                """
                return self.distance_function.prob(value=x, name='diff prob')

            def diff_cumulative_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
                f"""
                Derivative of cumulative function of the distribution
                :param x: An array with the size of (1xC)
                :return: derivatives of CDF of the distribution with respect to variable x calculated in size
                of (1xC)
                """
                return self.distance_function.cdf(value=x, name='diff cdf')

            def diff_log_probability_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
                f"""
                 Derivative of the  log of distribution
                :param x: An array with the size of (1xC)
                :return: derivatives of the log of the distribution with respect to variable x calculated in
                size of (1xC)
                """
                return self.distance_function.log_prob(value=x, name='diff log  prob')

            def diff_log_cumulative_distribution_(x: jnp.ndarray = None) -> jnp.ndarray:
                f"""
                Derivative of the log of cumulative function of the distribution
                :param x: An array with the size of (1xC)
                :return: derivatives of the log CDF of the distribution with respect to variable x calculated in
                size of (1xC)
                """
                return self.distance_function.log_cdf(value=x, name='diff log  cdf')

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        def mle(x: jnp.ndarray = None, checking_inputs: bool = False) -> dict:
            """
            Enter an array of the data to fit the distribution of parameters using MLE
            :param x: Nx1 array of Data
            :param checking_inputs: A boolean variable used to activate checking the correctness of input variables
            :return: A dictionary of the distribution parameters
            """
            return self.distance_function.experimental_fit(value=x, validate_args=checking_inputs).parameters

        def sampling_from_distribution(sample_shape: tuple = None) -> jnp.ndarray:
            return (self.distance_function.sample(sample_shape=sample_shape, seed=self.key, name='sample from pdf')).T

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        self.sample = sampling_from_distribution
        self.maximum_liklihood_estimation = mle

        if self.activate_jit:  # activating jit
            self.pdf = jit(vmap(fun=probability_distribution_, in_axes=self.in_vec_dim,
                                out_axes=self.out_vec_dim))

            self.log_pdf = jit(vmap(fun=log_probability_distribution_, in_axes=self.in_vec_dim,
                                    out_axes=self.out_vec_dim))

            self.cdf = jit(
                vmap(fun=cumulative_distribution_, in_axes=self.in_vec_dim, out_axes=self.out_vec_dim))

            self.log_cdf = jit(vmap(fun=log_cumulative_distribution_, in_axes=self.in_vec_dim,
                                    out_axes=self.out_vec_dim))

            self.diff_pdf = jit(vmap(jacfwd(fun=diff_probability_distribution_), in_axes=self.in_vec_dim
                                     , out_axes=self.out_vec_dim))

            self.diff_log_pdf = jit(vmap(jacfwd(fun=diff_log_probability_distribution_),
                                         in_axes=self.in_vec_dim, out_axes=self.out_vec_dim))

            self.diff_cdf = jit(vmap(jacfwd(fun=diff_cumulative_distribution_), in_axes=self.in_vec_dim
                                     , out_axes=self.out_vec_dim))

            self.diff_log_cdf = jit(vmap(jacfwd(fun=diff_log_cumulative_distribution_),
                                         in_axes=self.in_vec_dim, out_axes=self.out_vec_dim))

        else:  # Only using vectorized function
            self.pdf = vmap(fun=probability_distribution_, in_axes=self.in_vec_dim,
                            out_axes=self.out_vec_dim)

            self.log_pdf = vmap(fun=log_probability_distribution_, in_axes=self.in_vec_dim,
                                out_axes=self.out_vec_dim)

            self.cdf = vmap(fun=cumulative_distribution_, in_axes=self.in_vec_dim,
                            out_axes=self.out_vec_dim)

            self.log_cdf = vmap(fun=log_cumulative_distribution_, in_axes=self.in_vec_dim,
                                out_axes=self.out_vec_dim)

            self.diff_pdf = jit(vmap(jacfwd(fun=diff_probability_distribution_), in_axes=self.in_vec_dim
                                     , out_axes=self.out_vec_dim))

            self.diff_log_pdf = jit(vmap(jacfwd(fun=diff_log_probability_distribution_),
                                         in_axes=self.in_vec_dim, out_axes=self.out_vec_dim))

            self.diff_cdf = jit(vmap(jacfwd(fun=diff_cumulative_distribution_), in_axes=self.in_vec_dim
                                     , out_axes=self.out_vec_dim))

            self.diff_log_cdf = jit(vmap(jacfwd(fun=diff_log_cumulative_distribution_),
                                         in_axes=self.in_vec_dim, out_axes=self.out_vec_dim))


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class Uniform(ContinuousDistributions):
    def __init__(self, lower: jnp.ndarray = None, upper: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        In probability theory and statistics, the continuous uniform distribution or rectangular distribution is a
        family of symmetric probability distributions. The distribution describes an experiment where there is an
        arbitrary outcome that lies between certain bounds. The bounds are defined by the parameters, lower and upper,
        which are the minimum and maximum values.
        [1] Dekking, Michel (2005). A modern introduction to probability and statistics : understanding why and how.
        London, UK: Springer. pp. 60–61. ISBN 978-1-85233-896-1

        Continuous uniform distribution
        :param lower: A ndarray or float indicating the lower bound of the distribution
        :param upper: A ndarray or float indicating the upper bound of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(Uniform, self).__init__(lower=lower, upper=upper, activate_jit=activate_jit, random_seed=random_seed,
                                      n_chains=n_chains, in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim,
                                      validate_input_range=validate_input_range)
        self.name = 'Uniform'
        # checking the correctness of the parameters
        if jnp.any(self.lower >= self.upper):
            raise Exception(f'The lower limit of the uniform distribution is greater than the upper limit'
                            f' ({self.name} distribution)!')

        if not isinstance(self.lower, jnp.ndarray) or not isinstance(self.upper, jnp.ndarray):
            raise Exception(f'Please enter the input parameter in the format of ndarray ({self.__class__}'
                            f' distribution)')
        self.distance_function = distributions.Uniform(low=self.lower.tolist(), high=self.upper.tolist(),
                                                       name='Uniform')
        ContinuousDistributions.parallelization(self)

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Normal(ContinuousDistributions):
    def __init__(self, loc: jnp.ndarray = None, scale: jnp.ndarray = None,
                 var: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous Normal distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(Normal, self).__init__(loc=loc, scale=scale, var=var, activate_jit=activate_jit, random_seed=random_seed,
                                     n_chains=n_chains, in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim,
                                     validate_input_range=validate_input_range)
        self.name = 'Normal'
        # checking the correctness of the parameters
        if not isinstance(self.loc, type(self.scale)):
            raise Exception(f'The input parameters are not consistent ({self.name} distribution)!')

        if self.loc is None or self.scale is None:
            raise Exception(f'The value of either mean or standard deviation is not specified'
                            f'({self.name} distribution)!')

        self.distance_function = distributions.Normal(scale=self.scale.tolist(), loc=self.loc.tolist(),
                                                      name=self.name)

        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class TruncatedNormal(ContinuousDistributions):
    def __init__(self, loc: jnp.ndarray = None, var: jnp.ndarray = None,
                 scale: jnp.ndarray = None, lower: jnp.ndarray = None,
                 upper: jnp.ndarray = None, activate_jit: bool = False, random_seed: int = 1,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1,
                 validate_input_range: bool = True) -> None:
        """
        Continuous TruncatedNormal distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(TruncatedNormal, self).__init__(lower=lower, upper=upper, scale=scale, loc=loc, var=var,
                                              activate_jit=activate_jit, random_seed=random_seed,
                                              n_chains=n_chains, in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim,
                                              validate_input_range=validate_input_range)
        self.name = 'TruncatedNormal'
        # checking the correctness of the parameters

        if jnp.any(self.lower >= self.upper):
            raise Exception(f'The input lower bound cannot be greater than input upper bound '
                            f' ({self.name} distribution)!')

        self.distance_function = distributions.TruncatedNormal(loc=self.loc.tolist(), scale=self.scale.tolist(),
                                                               low=self.lower.tolist(), high=self.upper.tolist(),
                                                               name=self.name)

        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class HalfNormal(ContinuousDistributions):
    def __init__(self, scale: jnp.ndarray = None, var: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous HalfNormal distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(HalfNormal, self).__init__(scale=scale, var=var, activate_jit=activate_jit, random_seed=random_seed,
                                         n_chains=n_chains, in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim,
                                         validate_input_range=validate_input_range)
        self.name = 'HalfNormal'
        # checking the correctness of the parameters

        self.distance_function = distributions.HalfNormal(scale=self.scale.tolist(),
                                                          name=self.name)

        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class TwoPieceNormal(ContinuousDistributions):
    def __init__(self, alpha: jnp.ndarray = None, scale: jnp.ndarray = None, var: jnp.ndarray = None,
                 loc: jnp.ndarray = None, validate_input_range: bool = True,
                 activate_jit: bool = False, random_seed: int = 1,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous  distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(TwoPieceNormal, self).__init__(var=var, loc=loc, scale=scale, alpha=alpha,
                                             validate_input_range=validate_input_range,
                                             activate_jit=activate_jit, random_seed=random_seed,
                                             n_chains=n_chains, in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'TwoPieceNormal'
        # checking the correctness of the parameters

        self.distance_function = distributions.TwoPieceNormal(loc=self.loc.tolist(), scale=self.scale.tolist(),
                                                              skewness=self.alpha.tolist(), name=self.name)
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

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Beta(ContinuousDistributions):
    def __init__(self, alpha: jnp.ndarray = None, beta: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous  distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(Beta, self).__init__(alpha=alpha, beta=beta, activate_jit=activate_jit, random_seed=random_seed,
                                   n_chains=n_chains, in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim,
                                   validate_input_range=validate_input_range)
        self.name = 'Beta'

        self.distance_function = distributions.Beta(force_probs_to_zero_outside_support=True,
                                                    concentration0=self.beta.tolist(),
                                                    concentration1=self.alpha.tolist(),
                                                    name=self.name)

        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(a=x, a_min=0, a_max=1.0)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Kumaraswamy(ContinuousDistributions):
    def __init__(self, alpha: jnp.ndarray = None, beta: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous Kumaraswamy distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(Kumaraswamy, self).__init__(alpha=alpha, beta=beta, activate_jit=activate_jit, random_seed=random_seed,
                                          n_chains=n_chains, validate_input_range=validate_input_range,
                                          in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'Kumaraswamy'
        self.distance_function = distributions.Kumaraswamy(concentration1=self.alpha.tolist(),
                                                           concentration0=self.beta.tolist(),
                                                           allow_nan_stats=False,
                                                           validate_args=True,
                                                           name=self.name)
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=1.0 - jnp.finfo(float).eps)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Exponential(ContinuousDistributions):
    def __init__(self, rate: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous  distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(Exponential, self).__init__(rate=rate, activate_jit=activate_jit, random_seed=random_seed,
                                          n_chains=n_chains, validate_input_range=validate_input_range,
                                          in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'Exponential'
        self.distance_function = distributions.Exponential(rate=self.rate.tolist(),
                                                           force_probs_to_zero_outside_support=True,
                                                           name=self.name)
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance')
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(a=x, a_min=0, a_max=jnp.inf)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Laplace(ContinuousDistributions):
    def __init__(self, loc: jnp.ndarray = None, scale: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous Laplace distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(Laplace, self).__init__(loc=loc, scale=scale, activate_jit=activate_jit, random_seed=random_seed,
                                      n_chains=n_chains, validate_input_range=validate_input_range,
                                      in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'Laplace'
        self.distance_function = distributions.Laplace(scale=self.scale.tolist(), loc=self.loc.tolist(),
                                                       name=self.name)
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class StudentT(ContinuousDistributions):
    def __init__(self, loc: jnp.ndarray = None, scale: jnp.ndarray = None, df: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous StudentT distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(StudentT, self).__init__(loc=loc, scale=scale, df=df, activate_jit=activate_jit, random_seed=random_seed,
                                       n_chains=n_chains, validate_input_range=validate_input_range,
                                       in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'StudentT'
        self.distance_function = distributions.StudentT(loc=self.loc.tolist(), df=self.df.tolist(),
                                                        scale=self.scale.tolist(),
                                                        validate_args=True,
                                                        name=self.name)
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class HalfStudentT(ContinuousDistributions):
    def __init__(self, scale: jnp.ndarray = None, loc: jnp.ndarray = None, df: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous HalfStudentT distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(HalfStudentT, self).__init__(scale=scale, loc=loc, df=df, activate_jit=activate_jit,
                                           random_seed=random_seed,
                                           n_chains=n_chains, validate_input_range=validate_input_range,
                                           in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'HalfStudentT'
        self.distance_function = distributions.HalfStudentT(loc=self.loc.tolist(), scale=self.scale.tolist(),
                                                            df=self.df.tolist(),
                                                            validate_args=True,
                                                            name=self.name)
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(a=x, a_min=self.loc, a_max=jnp.inf)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Cauchy(ContinuousDistributions):
    def __init__(self, loc: jnp.ndarray = None, scale: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous Cauchy distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(Cauchy, self).__init__(loc=loc, scale=scale, activate_jit=activate_jit, random_seed=random_seed,
                                     n_chains=n_chains, validate_input_range=validate_input_range,
                                     in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'Cauchy'
        self.distance_function = distributions.Cauchy(loc=self.loc.tolist(), scale=self.scale.tolist(),
                                                      validate_args=True,
                                                      name=self.name)
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class HalfCauchy(ContinuousDistributions):
    def __init__(self, loc: jnp.ndarray = None, scale: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous  distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(HalfCauchy, self).__init__(loc=loc, scale=scale, activate_jit=activate_jit, random_seed=random_seed,
                                         n_chains=n_chains, validate_input_range=validate_input_range,
                                         in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'HalfCauchy'
        self.distance_function = distributions.HalfCauchy(loc=self.loc.tolist(), scale=self.scale.tolist(),
                                                          validate_args=True,
                                                          name=self.name)
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Gamma(ContinuousDistributions):
    def __init__(self, alpha: jnp.ndarray = None, beta: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous Gamma distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(Gamma, self).__init__(alpha=alpha, beta=beta, activate_jit=activate_jit, random_seed=random_seed,
                                    n_chains=n_chains, validate_input_range=validate_input_range,
                                    in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'Gamma'
        self.distance_function = distributions.Gamma(concentration=self.alpha.tolist(), rate=self.beta.tolist(),
                                                     validate_args=True,
                                                     name=self.name)
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=jnp.inf)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class InverseGamma(ContinuousDistributions):
    def __init__(self, alpha: jnp.ndarray = None, beta: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous InverseGamma distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(InverseGamma, self).__init__(beta=beta, alpha=alpha, activate_jit=activate_jit, random_seed=random_seed,
                                           n_chains=n_chains, validate_input_range=validate_input_range,
                                           in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'InverseGamma'
        self.distance_function = distributions.InverseGamma(concentration=self.alpha.tolist(), scale=self.beta.tolist(),
                                                            validate_args=True,
                                                            name=self.name)
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=jnp.inf)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Weibull(ContinuousDistributions):
    def __init__(self, kappa: jnp.ndarray = None, lambd: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous Weibull distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(Weibull, self).__init__(kappa=kappa, lambd=lambd, activate_jit=activate_jit, random_seed=random_seed,
                                      n_chains=n_chains, validate_input_range=validate_input_range,
                                      in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'Weibull'
        self.distance_function = distributions.Weibull(concentration=self.kappa.tolist(), scale=self.lambd.tolist(),
                                                       validate_args=True,
                                                       name=self.name)
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(a=x, a_min=0, a_max=jnp.inf)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Chi2(ContinuousDistributions):
    def __init__(self, df: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous  distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(Chi2, self).__init__(df=df, activate_jit=activate_jit, random_seed=random_seed,
                                   n_chains=n_chains, validate_input_range=validate_input_range,
                                   in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'Chi2'
        self.distance_function = distributions.Chi2(df=self.df.tolist(),
                                                    validate_args=True,
                                                    name=self.name)
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=jnp.inf)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class LogNormal(ContinuousDistributions):
    def __init__(self, loc: jnp.ndarray = None, scale: jnp.ndarray = None, var: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous  distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(LogNormal, self).__init__(loc=loc, scale=scale, var=var, activate_jit=activate_jit,
                                        random_seed=random_seed,
                                        n_chains=n_chains, validate_input_range=validate_input_range,
                                        in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'LogNormal'
        self.distance_function = distributions.LogNormal(loc=self.loc.tolist(), scale=self.scale.tolist(),
                                                         validate_args=True,
                                                         name=self.name)
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=jnp.inf)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Pareto(ContinuousDistributions):
    def __init__(self, alpha: jnp.ndarray = None, scale: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous Pareto distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(Pareto, self).__init__(scale=scale, alpha=alpha, activate_jit=activate_jit, random_seed=random_seed,
                                     n_chains=n_chains, validate_input_range=validate_input_range,
                                     in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'Pareto'
        self.distance_function = distributions.Pareto(concentration=self.alpha.tolist(), scale=self.scale.tolist(),
                                                      validate_args=True,
                                                      name=self.name)
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(a=x, a_min=self.scale, a_max=jnp.inf)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class ExponentiallyModifiedGaussian(ContinuousDistributions):
    def __init__(self, loc: jnp.ndarray = None, scale: jnp.ndarray = None, rate: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous ExponentiallyModifiedGaussian distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(ExponentiallyModifiedGaussian, self).__init__(loc=loc, scale=scale, rate=rate, activate_jit=activate_jit,
                                                            random_seed=random_seed,
                                                            n_chains=n_chains,
                                                            validate_input_range=validate_input_range,
                                                            in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'ExponentiallyModifiedGaussian'
        self.distance_function = distributions.ExponentiallyModifiedGaussian(loc=self.loc.tolist(),
                                                                             scale=self.scale.tolist(),
                                                                             rate=self.rate.tolist(),
                                                                             validate_args=True,
                                                                             name=self.name)
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Triangular(ContinuousDistributions):
    def __init__(self, lower: jnp.ndarray = None, upper: jnp.ndarray = None, peak: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous  distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(Triangular, self).__init__(lower=lower, upper=upper, peak=peak, activate_jit=activate_jit,
                                         random_seed=random_seed,
                                         n_chains=n_chains, validate_input_range=validate_input_range,
                                         in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'Triangular'
        self.distance_function = distributions.Triangular(low=self.lower.tolist(), high=self.upper.tolist(),
                                                          peak=self.peak.tolist(),
                                                          validate_args=True,
                                                          name=self.name)
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Gumbel(ContinuousDistributions):
    def __init__(self, loc: jnp.ndarray = None, scale: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous Gumbel distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(Gumbel, self).__init__(loc=loc, scale=scale, activate_jit=activate_jit, random_seed=random_seed,
                                     n_chains=n_chains, validate_input_range=validate_input_range,
                                     in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'Gumbel'
        self.distance_function = distributions.Gumbel(scale=self.scale.tolist(), loc=self.loc.tolist(),
                                                      validate_args=True,
                                                      name=self.name)
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Logistic(ContinuousDistributions):
    def __init__(self, loc: jnp.ndarray = None, scale: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous  distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(Logistic, self).__init__(loc=loc, scale=scale, activate_jit=activate_jit, random_seed=random_seed,
                                       n_chains=n_chains, validate_input_range=validate_input_range,
                                       in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'Logistic'
        self.distance_function = distributions.Logistic(loc=self.loc.tolist(), scale=self.scale.tolist(),
                                                        validate_args=True,
                                                        name=self.name)
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class LogitNormal(ContinuousDistributions):
    def __init__(self, loc: jnp.ndarray = None, scale: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous LogitNormal distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(LogitNormal, self).__init__(loc=loc, scale=scale, activate_jit=activate_jit, random_seed=random_seed,
                                          n_chains=n_chains, validate_input_range=validate_input_range,
                                          in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'LogitNormal'
        self.distance_function = distributions.LogitNormal(loc=self.loc.tolist(), scale=self.scale.tolist(),
                                                           validate_args=True,
                                                           name=self.name)
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information

    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return x
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class TruncatedCauchy(ContinuousDistributions):
    def __init__(self, loc: jnp.ndarray = None, scale: jnp.ndarray = None, lower: jnp.ndarray = None,
                 upper: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Continuous TruncatedCauchy distribution
        :param  : A ndarray or float indicating ----- of the distribution
        :param  : A ndarray or float indicating ---- of the distribution
        :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
        :param random_seed: An integer used to specify the random seed
        :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
         distribution with different parameters
        :param n_chains: An integer used to indicate the number of chains/samples
        :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
        :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
        """
        # recalling parameter values from the main parent class
        super(TruncatedCauchy, self).__init__(loc=loc, scale=scale, lower=lower, upper=upper, activate_jit=activate_jit, random_seed=random_seed,
                                  n_chains=n_chains, validate_input_range=validate_input_range,
                                  in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = 'TruncatedCauchy'
        self.distance_function = distributions.TruncatedCauchy(loc=self.loc.tolist(), scale=self.scale.tolist(),
                                                               low=self.lower.tolist(), high=self.upper.tolist(),
                                                               validate_args=True,
                                                                name=self.name)
        ContinuousDistributions.parallelization(self)

    @property
    def statistics(self):
        information = {'mean': self.distance_function.mean(name='mean'),
                       'entropy': self.distance_function.entropy(name='entropy'),
                       'mode': self.distance_function.mode(name='mode'),
                       'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
                       'median': self.distance_function.quantile(value=0.5, name='median'),
                       'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
                       'range': self.distance_function.range(name='range'),
                       'std': self.distance_function.stddev(name='stddev'),
                       'var': self.distance_function.variance(name='variance'),
                       }
        return information
    def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=1.0 - jnp.finfo(float).eps)


##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
x = random.uniform(key=random.PRNGKey(7), minval=-10, maxval=20, shape=(1, 1000), dtype=jnp.float64)
# KK = Uniform(lower=jnp.array([4]), upper=jnp.array([7]), activate_jit=True, random_seed=6,
#              in_vec_dim=1, out_vec_dim=1)


# KK = TruncatedNormal(scale=2, loc=7, lower=-4,
#                      upper=7, activate_jit=False, multi_distribution=False, in_vec_dim=1,
#                      out_vec_dim=1)
# KK = TruncatedNormal(scale=2, loc=3, lower=-2, upper=4, activate_jit=True,multi_distribution=False)
# KK = HalfNormal(scale=jnp.array([4]), activate_jit=True, multi_distribution=True)
# KK = TwoPieceNormal(scale=jnp.array([4, 5]), loc=jnp.array([4, 8]), alpha=jnp.array([4, 8]), activate_jit=False)
# KK = Beta(alpha=jnp.array([4, 5]), beta=jnp.array([4, 8]), activate_jit=False)
# KK = Kumaraswamy(alpha=jnp.array([4, 7]), beta=jnp.array([6, 9]), validate_input_range=True)
# KK = distributions.Kumaraswamy(concentration0=jnp.array([4,3]),concentration1=jnp.array([6,3]))

E1 = KK.pdf(x)
E2 = KK.log_pdf(x)
E4 = KK.cdf(x)
E5 = KK.log_cdf(x)
E6 = KK.diff_pdf(x)
E3 = KK.diff_log_pdf(x)

E8 = KK.diff_cdf(x)
E9 = KK.diff_log_cdf(x)
E11 = KK.statistics
E12 = KK.sample(100)
E10 = KK.maximum_liklihood_estimation(x=x)
E11
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# class pdf(ContinuousDistributions):
#     def __init__(self, : jnp.ndarray = None, : jnp.ndarray = None,
#                  activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
#                  n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
#         """
#         Continuous  distribution
#         :param  : A ndarray or float indicating ----- of the distribution
#         :param  : A ndarray or float indicating ---- of the distribution
#         :param activate_jit: A boolean variable used to activate/deactivate just-in-time evaluation
#         :param random_seed: An integer used to specify the random seed
#         :param multi_distribution: A boolean variable used to indicate the evaluation of multiple probability
#          distribution with different parameters
#         :param n_chains: An integer used to indicate the number of chains/samples
#         :param in_vec_dim: An integer used to indicate the axis of the input variable x for parallelized calculations
#         :param out_vec_dim: An integer used to indicate the axis of the output variable for exporting the output
#         """
#         # recalling parameter values from the main parent class
#         super(pdf, self).__init__(=, =, activate_jit=activate_jit, random_seed=random_seed,
#                                   n_chains=n_chains, validate_input_range=validate_input_range,
#                                   in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
#         self.name = ''
#         self.distance_function = distributions.(=self..tolist(), =self..tolist(),
#                                                            name=self.name)
#         ContinuousDistributions.parallelization(self)
#
#     @property
#     def statistics(self):
#         information = {'mean': self.distance_function.mean(name='mean'),
#                        'entropy': self.distance_function.entropy(name='entropy'),
#                        'mode': self.distance_function.mode(name='mode'),
#                        'first_quantile': self.distance_function.quantile(value=0.25, name='first quantile'),
#                        'median': self.distance_function.quantile(value=0.5, name='median'),
#                        'third_quantile': self.distance_function.quantile(value=0.75, name='third quantile'),
#                        'range': self.distance_function.range(name='range'),
#                        'std': self.distance_function.stddev(name='stddev'),
#                        'var': self.distance_function.variance(name='variance'),
#                        }
#         return information
#     def valid_range(self, x: jnp.ndarray) -> jnp.ndarray:
#         return jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=1.0 - jnp.finfo(float).eps)
