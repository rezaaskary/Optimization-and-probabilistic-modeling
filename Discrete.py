from tensorflow_probability.substrates.jax import distributions
from jax import vmap, jit, grad, random, lax, scipy, jacfwd
import jax.numpy as jnp


class DiscreteDistributions:
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
        if isinstance(n, jnp.ndarray):
            self.n = n
        elif n is None:
            self.n = None
        else:
            raise Exception(
                f'The value of input parameters is not specified correctly. Please enter parameters  in the'
                f' format of ndarrauy ({self.__class__} distribution)!')

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(p, jnp.ndarray):
            self.p = p
        elif p is None:
            self.p = None
        else:
            raise Exception(
                f'The value of input parameters is not specified correctly. Please enter parameters  in the'
                f' format of ndarrauy ({self.__class__} distribution)!')








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
            return (
                self.distance_function.sample(sample_shape=sample_shape, seed=self.key, name='sample from pdf')).T

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

class Binomial(DiscreteDistributions):
    def __init__(self, n: jnp.ndarray = None, p: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
                 n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
        """
        Discrete Binomial distribution
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
        super(Binomial, self).__init__(n=n, p=p, activate_jit=activate_jit, random_seed=random_seed,
                                  n_chains=n_chains, validate_input_range=validate_input_range,
                                  in_vec_dim=in_vec_dim, out_vec_dim=out_vec_dim)
        self.name = ''
        self.distance_function = distributions.Binomial(=self..tolist(), =self..tolist(),
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




# class pdf(DiscreteDistributions):
#     def __init__(self, : jnp.ndarray = None, : jnp.ndarray = None,
#                  activate_jit: bool = False, random_seed: int = 1, validate_input_range: bool = True,
#                  n_chains: int = 1, in_vec_dim: int = 1, out_vec_dim: int = 1) -> None:
#         """
#         Discrete  distribution
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
