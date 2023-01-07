import jax.numpy as jnp
from jax import lax, vmap, jit, random, scipy
from tensorflow_probability.substrates.jax import distributions


class DistanceNormilizer:
    def __init__(self, dists: list = None):
        self.ditance_list = jnp.zeros(shape=(6,), dtype=int)

        def _main_fcn_normalizer(index: int, normalization_parameters: tuple) -> jnp.ndarray:
            normalized_x = normalization_parameters
            normalized_x = normalized_x.at[:, index].set(jnp.where(self.dist_code[index] == 4,
                                        self._lognorm_(samples=self.x[:, index], bounds=self.bounds[index, :],
                                                                    loc_std=self.scale_loc[index, :]),
                jnp.where(self.dist_code[index] == 3, self._triangle_(samples=self.x[:, index],
                                                                      bounds=self.bounds[index, :],
                                                                      loc_std=self.scale_loc[index, :]),
                jnp.where(self.dist_code[index] == 2, self._truncnorm_(samples=self.x[:, index],
                                                                       bounds=self.bounds[index, :],
                                                                       loc_std=self.scale_loc[index, :]),
                jnp.where(self.dist_code[index] == 1, self._norm_(samples=self.x[:, index],
                                                                  bounds=self.bounds[index, :],
                                                                  loc_std=self.scale_loc[index, :]),
                          self._norm_(samples=self.x[:, index],
                                  bounds=self.bounds[index, :],
                                  loc_std=self.scale_loc[index, :]))))))
            return normalized_x

        self._main_fcn_normalizer = _main_fcn_normalizer

    def apply_normalization(self):
        normed_samples = lax.fori_loop(lower=0,
                      upper=self.num_vars,
                      body_fun=self._main_fcn_normalizer,
                      init_val=(jnp.zeros(jnp.zeros([self.n * self.num_vars, self.num_vars], dtype=jnp.float32)))

                      )

        return normed_samples

    def _uniform_(self, samples: jnp.ndarray = None, bounds: jnp.ndarray = None, loc_std: jnp.ndarray = None):
        return samples * (bounds[1] - bounds[0]) + bounds[0]

    def _triangle_(self, samples: jnp.ndarray = None, bounds: jnp.ndarray = None, loc_std: jnp.ndarray = None):
        samples = jnp.clip(a=samples, a_min=0.0, a_max=1.0)
        mid = (loc_std[0] - bounds[0]) / (bounds[1] - bounds[0])
        normed = jnp.where(samples < mid,
                           bounds[0] + (samples * (bounds[1] - bounds[0]) * (loc_std[0] - bounds[0])) ** 0.5,
                           bounds[1] - ((bounds[1] - loc_std[0]) ** 2 - (samples * (bounds[1] - bounds[0]) *
                                                                         (bounds[1] - loc_std[0]) -
                                                                         (bounds[1] - loc_std[0]) * (
                                                                                 loc_std[0] - bounds[0]))) ** 0.5)
        return jnp.clip(a=normed, a_min=bounds[0], a_max=bounds[1])

    def _norm_(self, samples: jnp.ndarray = None, bounds: jnp.ndarray = None, loc_std: jnp.ndarray = None):
        return scipy.stats.norm.ppf(q=samples, loc=loc_std[0], scale=loc_std[1])

    # (params[:, i], loc=b1, scale=b2)
    def _truncnorm_(self, samples: jnp.ndarray = None, bounds: jnp.ndarray = None, loc_std: jnp.ndarray = None):
        return distributions.TruncatedNormal(loc=loc_std[0], scale=loc_std[1], low=bounds[0], high=bounds[1]).quantile(
            samples)

    def _lognorm_(self, samples: jnp.ndarray = None, bounds: jnp.ndarray = None, loc_std: jnp.ndarray = None):
        return jnp.exp(scipy.stats.norm.ppf(q=samples, loc=loc_std[0], scale=loc_std[1]))


class FourierAmplitudeSensitivityTest(DistanceNormilizer):
    def __init__(self,
                 num_vars: int = None,
                 bounds: jnp.ndarray = None,
                 scale_loc: list = None,
                 names: list = None,
                 dists: list = None,
                 groups: list = None,
                 n: int = None,
                 terms: int = None,
                 seed: int = None):

        if isinstance(num_vars, int):
            self.num_vars = num_vars
        else:
            raise Exception('The number of variable is not entered')

        if isinstance(scale_loc, list):
            if len(scale_loc) != self.num_vars:
                raise Exception('The list of scale/location of parameters is not consistent with the number\n'
                                'of parameters')
        else:
            raise Exception('The scale/location parameters should be entered in the format of a list.')

        if isinstance(bounds, jnp.ndarray):
            self.bounds = jnp.array(bounds, dtype=jnp.float32).reshape((-1, 2))
        else:
            raise Exception('The values for the lower/upper bounds of parameters are not specified correctly')

        if jnp.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError(f'The values of lower bound cannot be larger than the upper bounds.')

        self.scale_loc = jnp.zeros(shape=(self.num_vars, 2), dtype=jnp.float32)
        self.scale_loc = self.scale_loc.at[:, :].set(jnp.nan)
        self.dist_code = jnp.zeros(shape=(self.num_vars,), dtype=int)
        self.dists = []

        if isinstance(dists, list):
            for cntr, item in enumerate(dists):
                if item == 'norm':
                    self.dists.append('norm')
                    self.dist_code = self.dist_code.at[cntr].set(1)

                    if len(scale_loc[cntr]) == 2 and scale_loc[cntr][1] > 0:
                        self.scale_loc = self.scale_loc.at[cntr, 1].set(scale_loc[cntr][1])
                        self.scale_loc = self.scale_loc.at[cntr, 0].set(scale_loc[cntr][0])
                    elif len(scale_loc[cntr]) != 2:
                        raise Exception(f'Either standard deviation or mean value for {cntr}th parameter\n'
                                        f'is not specified')
                    else:
                        raise ValueError(f'The value of standard deviation entered for {cntr}th parameter\n'
                                         f'is a negative value.')

                elif item == 'lognorm':
                    self.dists.append('lognorm')
                    self.dist_code = self.dist_code.at[cntr].set(4)
                    if len(scale_loc[cntr]) == 2 and scale_loc[cntr][1] > 0:
                        self.scale_loc = self.scale_loc.at[cntr, 1].set(scale_loc[cntr][1])
                        self.scale_loc = self.scale_loc.at[cntr, 0].set(scale_loc[cntr][0])
                    elif len(scale_loc[cntr]) != 2:
                        raise Exception(f'Either standard deviation or mean value for {cntr}th parameter\n'
                                        f'is not specified')
                    else:
                        raise ValueError(f'The value of standard deviation entered for {cntr}th parameter\n'
                                         f'is a negative value.')

                elif item == 'truncnorm':
                    self.dists.append('truncnorm')
                    self.dist_code = self.dist_code.at[cntr].set(2)
                    if len(scale_loc[cntr]) == 2 and scale_loc[cntr][1] > 0:
                        self.scale_loc = self.scale_loc.at[cntr, 1].set(scale_loc[cntr][1])
                        self.scale_loc = self.scale_loc.at[cntr, 0].set(scale_loc[cntr][0])
                    elif len(scale_loc[cntr]) != 2:
                        raise Exception(f'Either standard deviation or mean value for {cntr}th parameter\n'
                                        f'is not specified')
                    else:
                        raise ValueError(f'The value of standard deviation entered for {cntr}th parameter\n'
                                         f'is a negative value.')
                elif item == 'triang':
                    self.dists.append('triang')
                    self.dist_code = self.dist_code.at[cntr].set(3)

                    if self.bounds[cntr, 0] < 0 or self.bounds[cntr, 1] >= 1:
                        raise ValueError('The basis (bounds) values of triangular distribution should be within\n'
                                         ' interval [0, 1].')

                    if len(scale_loc[cntr]) == 1 and (scale_loc[cntr][0] > self.bounds[cntr, 0]) and (
                            scale_loc[cntr][0] < self.bounds[cntr, 1]):
                        self.scale_loc = self.scale_loc.at[cntr, 0].set(scale_loc[cntr][0])
                    elif len(scale_loc[cntr]) != 1:
                        raise Exception(
                            f'The value of peak for triangular distribution in {cntr}th parameters is not \n'
                            f'specified.')
                    else:
                        raise ValueError(f'The value of lower/upper basis and the location of peak in the triangular\n'
                                         f' distribution for the {cntr}th parameter is not specified correctly')
                elif item == 'unif':
                    self.dists.append('unif')
                    self.dist_code = self.dist_code.at[cntr].set(0)
                else:
                    self.dists.append('unif')
                    self.dist_code = self.dist_code.at[cntr].set(0)
                    print(f'------------------------------------------------------------------------------\n'
                          f'The distribution function of the {cntr}th variable is not specified correctly.\n'
                          f'Therefore, uniform distribution is assigned to the {cntr}th variable.\n'
                          f'------------------------------------------------------------------------------')

        if self.num_vars != self.bounds.shape[0]:
            raise Exception('The number of variables is not consistent with the lower/upper bounds')

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

        DistanceNormilizer(dists=self.dist_code).apply_normalization(self)




        return self.x


class ExtendedFourierAmplitudeSensitivityTest:
    def __init__(self,
                 num_vars: int = None,
                 bounds: jnp.ndarray = None,
                 scale_loc: list = None,
                 names: list = None,
                 dists: list = None,
                 groups: list = None,
                 n: int = None,
                 seed: int = None):













problem = {
    'names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6'],
    'num_vars': 6,
    'bounds': jnp.array([[-jnp.pi, jnp.pi], [-.1, 0.2], [0.5, 0.75], [0.5, 0.75], [0.5, 0.75], [0.5, 0.75]]),
    'groups': ['G1', 'G2', 'G1', 'G1', 'G1', 'G1'],
    'dists': ['unif', 'lognorm', 'triang', 'triang', 'triang', 'triang']
}

scale_loc = [[], [0.5, 1.8], [0.6], [0.6], [0.6], [0.6]]
MM = FourierAmplitudeSensitivityTest(n=2048, terms=5, seed=3, num_vars=6, bounds=problem['bounds'],
                                     dists=problem['dists'], groups=problem['groups'], names=problem['names'],
                                     scale_loc=scale_loc)
RD = MM.solve()
import numpy as np

RD = np.array(RD)
RD
