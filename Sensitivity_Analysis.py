import jax.numpy as jnp
from jax import lax, vmap, jit, random

bounds = jnp.array([[-jnp.pi, jnp.pi], [1.0, 0.2], [3, 0.5]], dtype=jnp.float32)

problem = {
    'names': ['x1', 'x2', 'x3'],
    'num_vars': 3,
    'bounds': bounds,
    'groups': ['G1', 'G2', 'G1'],
    'dists': ['unif', 'lognorm', 'triang']
}


class DistanceNormilizer:
    def __init__(self, dists: list = None):
        self.ditance_list = jnp.zeros(shape=(6,),dtype=object)


    def _uniform_(self, parameter: jnp.ndarray = None, lb: jnp.ndarray = None, ub: jnp.ndarray = None):
        return parameter * (ub - lb) + lb

    def _triangle_(self):
        return

    def _norm_(self):
        return

    def _truncnorm_(self):
        return

    def _lognorm_(self):
        return


class FourierAmplitudeSensitivityTest(DistanceNormilizer):
    def __init__(self,
                 lb: jnp.ndarray = None,
                 ub: jnp.ndarray = None,
                 names: list = None,
                 dists: list = None,
                 groups: list = None,
                 n: int = None,
                 terms: int = None,
                 seed: int = None):

        if isinstance(lb, jnp.ndarray):
            self.lb = jnp.array(lb, dtype=jnp.float32).reshape((-1, 1))
        else:
            raise Exception('The values for the lower bound of parameters are not specified correctly')

        if isinstance(ub, jnp.ndarray):
            self.ub = jnp.array(ub, dtype=jnp.float32).reshape((-1, 1))
        else:
            raise Exception('The values for the upper bound of parameters are not specified correctly')

        if jnp.any(self.lb >= self.ub):
            raise ValueError('The upper and lower bounds are poorly defined. The values of lower bound cannot\n'
                             'be greater than upper bound.')

        self.num_vars = self.lb.shape[0]

        if isinstance(names, list):
            self.names = names
        elif names is None:
            self.names = ['x' + str(i) for i in range(self.num_vars)]
        else:
            raise ValueError('Please enter the list of names of parameters is string format.')

        if isinstance(groups, list):
            self.groups = groups
        elif groups is None:
            self.groups = ['g' + str(i) for i in range(self.num_vars)]
        else:
            raise ValueError('Please enter the list of names of parameters is string format.')

        if isinstance(dists, list):
            self.dists = dists
        elif dists is None:
            self.dists = ['uniform'] * self.num_vars
        else:
            raise Exception('Please correctly enter the list of distance variables.')

        if isinstance(n, int):
            self.n = n
        else:
            raise Exception('The number samples is not specified correctly!')

        if isinstance(terms, int):
            self.terms = terms
        else:
            raise Exception('The number of terms used for calculating the Fourier transformation')

        if isinstance(seed, int):
            self.key = random.PRNGKey(seed)
        else:
            raise Exception('Please enter an integer value to fix the random number generator')

        if self.n <= 4 * terms ** 2:
            raise Exception('Sample size n > 4terms^2 is required!')

        self.omega = jnp.zeros([self.num_vars])
        self.omega2 = self.omega.copy()
        self.omega = self.omega.at[0].set(jnp.floor((self.n - 1) / (2 * self.terms)))
        self.m = jnp.floor(self.omega[0] / (2 * self.terms))
        self.omega = self.omega.at[1:self.num_vars].set(jnp.where(self.m >= self.num_vars - 1,
                                                                  jnp.floor(jnp.linspace(1, self.m, self.num_vars - 1)),
                                                                  jnp.arange(self.num_vars - 1) % self.m + 1))
        self.s = (2 * jnp.pi / self.n) * jnp.arange(self.n)
        self.x = jnp.zeros([self.n * self.num_vars, self.num_vars])
        self.z = jnp.arange(start=0, stop=self.n, dtype=jnp.int32)
        self.idx = jnp.arange(start=1, stop=self.num_vars, dtype=jnp.int32)
        self.phi_rng_uniform = random.uniform(key=self.key, shape=(self.num_vars,), dtype=jnp.float32, maxval=1.0,
                                              minval=0)

    def solve(self):

        def _phase_shift_inner(j: int, values_2: tuple) -> tuple:
            omega2, z_idx, x_arg, phi_arg = values_2
            x_arg = x_arg.at[z_idx, j].set(0.5 + (1 / jnp.pi) * jnp.arcsin(jnp.sin(omega2[j] * self.s + phi_arg)))
            return omega2, z_idx, x_arg, phi_arg

        def _phase_shift(i: int, values_1: tuple) -> tuple:
            omega2, idx_new, x_arg = values_1
            omega2 = omega2.at[i].set(self.omega[0])
            idx_new = jnp.where(i == 0, idx_new, idx_new.at[i - 1].set(idx_new[i - 1] - 1))
            omega2 = omega2.at[idx_new].set(self.omega[1:self.num_vars])
            z_idx = self.z + i * self.n
            phi = 2 * jnp.pi * self.phi_rng_uniform[i]
            omega2, z_idx_, x_arg, phi_arg = lax.fori_loop(lower=0, upper=self.num_vars, body_fun=_phase_shift_inner,
                                                           init_val=(omega2, z_idx, x_arg, phi))
            return omega2, idx_new, x_arg

        self.omega2, idx_new, self.x = lax.fori_loop(lower=0, upper=self.num_vars, body_fun=_phase_shift,
                                                     init_val=(self.omega2, self.idx, self.x))
        return self.x


problem = {
    'names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6'],
    'num_vars': 6,
    'bounds': jnp.array([[-jnp.pi, jnp.pi], [1.0, 0.2], [3, 0.5], [3, 0.5], [3, 0.5], [3, 0.5]]),
    'groups': ['G1', 'G2', 'G1', 'G1', 'G1', 'G1'],
    'dists': ['unif', 'lognorm', 'triang', 'triang', 'triang', 'triang']
}

MM = FourierAmplitudeSensitivityTest(problem=problem, n=2048, terms=5, seed=3)
RD = MM.solve()
import numpy as np

RD = np.array(RD)
RD
