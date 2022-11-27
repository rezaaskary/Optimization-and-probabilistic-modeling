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