import sklearn.decomposition
from tensorflow_probability.substrates.jax import distributions
from jax import vmap, jit, grad, random, lax, scipy, jacfwd
import jax.numpy as jnp
import sys
import warnings


class PPCA:
    def __init__(self, y: jnp.ndarray = None, n_comp: int = 2, max_iter: int = 500, tolerance: float = 1e-6):
        if not (min(y.shape) > 1):
            raise Exception('Too few feature variables')

        if isinstance(tolerance, float):
            self.tolerance = tolerance
        else:
            raise Exception('The value of tolerance  should be a small float number.')

        if isinstance(max_iter, int):
            self.max_iter = max_iter
        else:
            raise Exception('Maximum number of iteration must be a positive integer.')

        if not isinstance(y, jnp.ndarray):
            raise Exception('Invalid format of input matrix y!. Please enter the input matrix with ndarray format')
        else:
            self.y = jnp.array(object=y.T, dtype=jnp.float64)

        self.nans = jnp.isnan(self.y)
        # finding features that for all samples are not measured
        all_nans_row_wise = jnp.all(self.nans, axis=1)
        if jnp.any(all_nans_row_wise):
            self.y = self.y[~all_nans_row_wise, :]
            self.nans = self.nans[~all_nans_row_wise, :]
            print(f'---------------------------------------------------------------------------------------------\n'
                  f'The {int(all_nans_row_wise.sum())} columns of the input matrix has no measurements.\n'
                  f'The matrix is shrank to {int(jnp.sum(~all_nans_row_wise))} columns.\n'
                  f'---------------------------------------------------------------------------------------------')

        # finding measurements with all features that not measured (rows with all NaN values)
        all_nans = jnp.all(self.nans, axis=0)
        if jnp.any(all_nans):
            self.y = self.y[:, ~all_nans]
            self.nans = self.nans[:, ~all_nans]
            print(f'---------------------------------------------------------------------------------------------\n'
                  f'The {int(all_nans.sum())} rows of the input matrix have no measurements.\n'
                  f'The matrix is shrank to {int(jnp.sum(~all_nans))} rows.\n'
                  f'---------------------------------------------------------------------------------------------')
        self.any_missing = jnp.any(self.nans)
        self.obs = ~self.nans
        self.num_obs = self.obs.sum()
        self.p, self.n = self.y.shape
        self.max_rank = min([self.p, self.n])

        if (not jnp.isscalar(n_comp)) or (not isinstance(n_comp, int)):
            raise Exception('invalid n_comp!. Please enter a scalar integer as the number of components')
        else:
            self.n_comp = n_comp

        if self.n_comp > self.max_rank - 1:
            self.n_comp = max([1, self.max_rank - 1])
            print(
                f'Warning: Maximum possible rank of the data is {elf.max_rank}. Computation continues with the number\n'
                f'of principal components k set to {self.n_comp}')

        self.w = random.normal(key=random.PRNGKey(1), shape=(self.p, self.n_comp), dtype=jnp.float64)
        self.v = random.uniform(key=random.PRNGKey(1), shape=(1, 1), dtype=jnp.float64)
        if not sys.warnoptions:
            warnings.simplefilter('ignore')

        self.mu = jnp.zeros(shape=(self.p, 1), dtype=jnp.float64)
        self.x = jnp.zeros(shape=(self.n_comp, self.n), dtype=jnp.float64)
        self.wnew = jnp.zeros(shape=(self.p, self.n_comp), dtype=jnp.float64)
        self.c = jnp.zeros(shape=(self.n_comp, self.n_comp, self.n), dtype=jnp.float64)
        self.nloglk = jnp.array(jnp.inf, dtype=jnp.float64)
        self.itr = 0.0
        self.delta = jnp.array(jnp.inf, dtype=jnp.float64).reshape((1, 1))
        self.diff = jnp.array(jnp.inf, dtype=jnp.float64)
        if self.any_missing:
            self.run = self._incomplete_matrix_cal
        else:
            self.run = self._complete_matrix_cal

    def _complete_matrix_cal(self):
        self.mu = jnp.mean(self.y, axis=1)[:, jnp.newaxis]
        self.y -= jnp.tile(self.mu, [1, self.n])
        self.traces = ((self.y.reshape((-1, 1))).T @ self.y.reshape((-1, 1))) / (self.n - 1)
        self.eps = jnp.finfo(float).eps

        def body_fun(value: tuple = None) -> tuple:
            itr, v, w, nloglk, delta, diff = value
            itr += 1
            sw = self.y @ (self.y.T @ w) / (self.n - 1)
            m = w.T @ w + v * jnp.eye(self.n_comp)
            wnew = sw @ jnp.linalg.inv(v * jnp.eye(self.n_comp) + jnp.linalg.solve(a=m, b=w.T) @ sw)
            vnew = (self.traces - jnp.trace(sw @ jnp.linalg.solve(a=m, b=wnew.T))) / self.p
            dw = (jnp.abs(w - wnew) / (jnp.sqrt(self.eps) + (jnp.abs(wnew)).max())).max()
            dv = jnp.abs(v - vnew) / (self.eps + v)
            delta = jnp.maximum(dw, dv)
            cc = wnew @ wnew.T + vnew * jnp.eye(self.p)
            nloglk_new = (self.p * jnp.log(2 * jnp.pi) + jnp.log(jnp.linalg.det(cc)) +
                          jnp.trace(jnp.linalg.inv(cc) @ self.y @ self.y.T / (self.n - 1))) * self.n / 2
            return itr, vnew, wnew, nloglk_new, delta, nloglk - nloglk_new

        def cond_fun(value: tuple = None):
            itr, v, w, nloglk, delta, diff = value
            arg4 = jnp.abs(diff).astype(float)
            return (arg4 > self.tolerance) | (itr < self.max_iter)

        self.itr, \
        self.v, \
        self.w, \
        self.nloglk, \
        self.delta, \
        self.diff = lax.while_loop(cond_fun=cond_fun, body_fun=body_fun,
                                   init_val=(self.itr,
                                             self.v,
                                             self.w,
                                             self.nloglk,
                                             self.delta,
                                             self.diff,
                                             ))
        m = self.w.T @ self.w + self.v * jnp.eye(self.n_comp)
        xmu = jnp.linalg.solve(a=m, b=self.w.T) @ self.y
        return self.w, xmu, self.mu, self.v, self.itr, self.nloglk

    def _incomplete_matrix_cal(self):

        while self.itr < self.max_iter:
            for j in range(self.n):
                ysamp = self.y[:, j:j + 1]
                idxobs = self.obs[:, j]
                wsamp = self.w[idxobs, :]
                cj = jnp.eye(self.n_comp) / self.v - (wsamp.T @ wsamp) @ jnp.linalg.inv(
                    jnp.eye(self.n_comp) + (wsamp.T @ wsamp) / self.v) / (self.v ** 2)
                self.x = self.x.at[:, j:j + 1].set(cj @ (wsamp.T @ (ysamp[idxobs] - self.mu[idxobs])))
                self.c = self.c.at[:, :, j].set(cj)

            self.mu = jnp.nanmean(self.y - self.w @ self.x, axis=1)[:, jnp.newaxis]

            for i in range(self.p):
                idxobs = self.obs[i, :]

                m = self.x[:, idxobs] @ self.x[:, idxobs].T + self.v * jnp.sum(self.c[:, :, idxobs], axis=2)
                wm = self.x[:, idxobs] @ (self.y[i, idxobs] - self.mu[i, 0]).T
                self.wnew = self.wnew.at[i, :].set(jnp.linalg.solve(m, wm))

            vsum = jnp.zeros((1, 1))

            for j in range(self.n):
                idxobs = self.obs[:, j]
                wnew_sample = self.wnew[idxobs, :]
                vsum = vsum + ((self.y[idxobs, j] - wnew_sample @ self.x[:, j] - self.mu[idxobs, 0]) ** 2 +
                               self.v * (jnp.diag(wnew_sample @ self.c[:, :, j] @ wnew_sample.T))).sum()

            self.vnew = vsum / self.num_obs
            nloglk_new = 0

            for j in range(self.n):
                idxobs = self.obs[:, j]
                y_c = self.y[idxobs, j:j + 1] - self.mu[idxobs, 0:1]

                wobs = self.wnew[idxobs, :]
                cy = wobs @ wobs.T + self.vnew * jnp.eye(idxobs.sum())
                nloglk_new = nloglk_new + (idxobs.sum() * jnp.log(2 * jnp.pi) + jnp.log(jnp.linalg.det(cy)) +
                                           jnp.trace(jnp.linalg.inv(cy) @ y_c @ y_c.T)) / 2

            dw = (jnp.abs(self.w - self.wnew) / (jnp.sqrt(eps) + (jnp.abs(self.wnew)).max())).max()

            self.w = self.wnew
            self.v = self.vnew
            print(jnp.abs(self.nloglk - nloglk_new))
            if jnp.abs(self.nloglk - nloglk_new) < self.tolerance:
                break

            self.nloglk = nloglk_new
        mux = self.x.mean(axis=1)[:, jnp.newaxis]
        self.x -= jnp.tile(mux, [1, self.n])
        self.mu += self.w @ mux
        return self.w, xmu, self.mu, self.v, self.itr, self.nloglk


# if __name__ == '__main__':
#     data = random.gamma(key=random.PRNGKey(23), a=0.2, shape=(5000, 5))
#     # data = data.at[4, 2].set(jnp.nan)
#     D = PPCA(y=data, n_comp=2, max_iter=500, tolerance=1e-5)
#     D.run()



import numpy as np
data = random.gamma(key=random.PRNGKey(23), a=0.2, shape=(5000, 5))
data = np.array(data)

D = sklearn.decomposition.FactorAnalysis(n_components=2).fit(data)
D


#
# class FactorAnalysis:
#     def __init__(self, x:jnp.ndarray = None, n_comp: int = None, tolerance: float = 1e-6, ):
#
#

