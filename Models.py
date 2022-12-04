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
                f'Warning: Maximum possible rank of the data is {self.max_rank}. Computation continues with the number\n'
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
        self.eps = jnp.finfo(float).eps
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

            dw = (jnp.abs(self.w - self.wnew) / (jnp.sqrt(self.eps) + (jnp.abs(self.wnew)).max())).max()

            self.w = self.wnew
            self.v = self.vnew
            print(jnp.abs(self.nloglk - nloglk_new))
            if jnp.abs(self.nloglk - nloglk_new) < self.tolerance:
                break

            self.nloglk = nloglk_new
        mux = self.x.mean(axis=1)[:, jnp.newaxis]
        self.x -= jnp.tile(mux, [1, self.n])
        self.mu += self.w @ mux
        return self.w, mux, self.mu, self.v, self.itr, self.nloglk


# if __name__ == '__main__':
#     data = random.gamma(key=random.PRNGKey(23), a=0.2, shape=(5000, 5))
#     # data = data.at[4, 2].set(jnp.nan)
#     D = PPCA(y=data, n_comp=2, max_iter=500, tolerance=1e-5)
#     D.run()


class FactorAnalysis:
    def __init__(self,
                 x: jnp.ndarray = None,
                 n_comp: int = None,
                 tolerance: float = 1e-8,
                 max_iter: int = 1000,
                 random_seed: int = 1,
                 method: str = 'EM') -> None:
        """

        :param x:
        :param n_comp:
        :param tolerance:
        :param max_iter:
        :param random_seed:
        :param method:
        """
        if isinstance(method, str) and method in ['sgd', 'EM']:
            self.method = method
        elif not method:
            raise Exception('Please enter the method of calculating the latent variables.')
        else:
            raise Exception('Please select from either svd or EM as the supported method for calculating the latent '
                            'variables.')

        if isinstance(random_seed, int):
            self.key = random.PRNGKey(random_seed)
        elif not random_seed:
            self.key = random.PRNGKey(1)
        else:
            raise Exception('Enter an integer as the value of seed for generating pseudo random numbers.')

        if isinstance(x, jnp.ndarray):
            if jnp.any(jnp.isnan(x)):
                raise Exception(f'There are NaN values in the input matrix!')
            else:
                self.x = x
                self.n, self.p = self.x.shape
                self.mean = self.x.mean(axis=0)
                self.var = self.x.var(axis=0)
                self.x_m = (self.x - jnp.tile(self.mean, reps=(self.n, 1))).T
        else:
            raise Exception(f'The format of {type(x)} is not supported.\n'
                            f'The input matrix should be given in ndarray format.')

        if isinstance(n_comp, int):
            if n_comp < 1:
                raise Exception('The minimum number of principal components should be a positive integer.')
            else:
                self.n_comp = n_comp
        elif not n_comp:
            self.n_comp = self.p
        else:
            raise Exception('The format of the number of component is not supported.\n'
                            ' Please enter the number of components as a positive integer!')

        if isinstance(tolerance, float):
            if tolerance > 1:
                raise Exception('Please enter a small value for tolerance. Ex. 1e-6')
            else:
                self.tolerance = tolerance
        elif not tolerance:
            self.tolerance = 1e-8
        else:
            raise Exception('The format of tolerance is not supported.\n'
                            ' Please enter a small value as tolerance (Ex. 1e-8)')

        if isinstance(max_iter, int):
            if max_iter < 1:
                raise Exception('Please enter a positive integer as the maximum number of iterations.')
            else:
                self.max_iter = max_iter
        elif not max_iter:
            self.max_iter = 1000
        else:
            raise Exception('The format of maximum iterations is not supported.\n'
                            ' Please enter positive integer as maximum number of iterations (Ex. 1000)')
        self.eps = jnp.finfo(float).eps
        self.itr = 0.0
        self.psi = random.uniform(key=self.key,
                                  shape=(self.p,),
                                  minval=0,
                                  maxval=1)

        self.f = random.uniform(key=self.key,
                                shape=(self.p, self.n_comp),
                                minval=0,
                                maxval=1)

        def _cond_fun(values: tuple = None) -> bool:
            itr, _, _, _, likelihood_error = values
            error = jnp.abs(likelihood_error).astype(float)
            return (error > self.tolerance) | (itr > self.max_iter)

        def _em_factor_analysis(values: tuple = None) -> tuple:
            itr, psi, f, old_log_likelihood, log_likelihood_error = values
            x_hat = jnp.diag(psi ** -0.5) @ self.x_m / jnp.sqrt(self.n)
            u_svd, s_svd, _ = jnp.linalg.svd(x_hat, full_matrices=False)
            a_svd = s_svd ** 2
            f = jnp.diag(psi ** 0.5) @ u_svd[:, :self.n_comp] @ jnp.diag(
                jnp.maximum(a_svd[:self.n_comp] - 1.0, self.eps) ** 0.5)
            likelihood = -0.5 * self.n * (jnp.log(a_svd[:self.n_comp]).sum() +
                                          self.n_comp + (a_svd[self.n_comp:]).sum() + jnp.log(
                        jnp.linalg.det(jnp.diag(psi * 2 * jnp.pi))))
            psi = self.var - jnp.diag(f @ f.T)
            log_likelihood_error = likelihood - old_log_likelihood
            itr += 1
            return itr, psi, f, likelihood, log_likelihood_error

        if self.method == 'EM':
            self.body_fun = _em_factor_analysis
            self.cond_fun = _cond_fun

    def fit(self):

        self.itr, \
            self.psi, self.f, \
            self.log_likelihood, \
            self.log_likelihood_error = lax.while_loop(body_fun=self.body_fun, cond_fun=self.cond_fun,
                                                       init_val=(self.itr,
                                                                 self.psi,
                                                                 self.f,
                                                                 jnp.array(-jnp.inf, dtype=jnp.float32),
                                                                 jnp.array(-jnp.inf, dtype=jnp.float32)))
        self.covariance = self.f @ self.f.T + jnp.diag(self.psi)

        return self

    def fit_transform(self):
        self.fit()
        coef = self.f / self.psi[:, jnp.newaxis]
        return jnp.linalg.inv(coef.T @ self.f + jnp.eye(self.n_comp)) @ coef.T @ self.x_m

class CanonicalCorrelationAnalysis:
    def __init__(self,
                 x: jnp.ndarray = None,
                 y: jnp.ndarray = None,
                 n_comp: int = None,
                 tolerance: float = 1e-8,
                 max_iter: int = 1000,
                 random_seed: int = 1,
                 method: str = 'EM') -> None:
        """

        :param x:
        :param y:
        :param n_comp:
        :param tolerance:
        :param max_iter:
        :param random_seed:
        :param method:
        """
        if isinstance(method, str) and method in ['sgd', 'EM']:
            self.method = method
        elif not method:
            raise Exception('Please enter the method of calculating the latent variables.')
        else:
            raise Exception('Please select from either svd or EM as the supported method for calculating the latent '
                            'variables.')

        if isinstance(random_seed, int):
            self.key = random.PRNGKey(random_seed)
        elif not random_seed:
            self.key = random.PRNGKey(1)
        else:
            raise Exception('Enter an integer as the value of seed for generating pseudo random numbers.')

        if isinstance(y, jnp.ndarray):
            if jnp.any(jnp.isnan(y)):
                raise Exception(f'There are NaN values in the input matrix y!')
            else:
                self.y = y
                self.n_y, self.p_y = self.y.shape
        else:
            raise Exception(f'The format of {type(y)} is not supported.\n'
                            f'The input matrix should be given in ndarray format.')

        if isinstance(x, jnp.ndarray):
            if jnp.any(jnp.isnan(x)):
                raise Exception(f'There are NaN values in the input matrix!')
            else:
                self.x = x
                self.n_x, self.p_x = self.x.shape
        else:
            raise Exception(f'The format of {type(x)} is not supported.\n'
                            f'The input matrix should be given in ndarray format.')

        if self.n_y == self.n_x:
            self.p_z = self.p_y + self.p_x
            self.z = jnp.concatenate(arrays=(self.x, self.y), axis=1)
            self.mean = self.z.mean(axis=0)
            self.var = self.z.var(axis=0)
            self.z_m = (self.z - jnp.tile(self.mean, reps=(self.n_x, 1))).T

        else:
            raise Exception('Matrices x and y have different observations. They are not consistent.')

        if isinstance(n_comp, int):
            if n_comp < 1:
                raise Exception('The minimum number of principal components should be a positive integer.')
            else:
                self.n_comp = n_comp
        elif not n_comp:
            self.n_comp = min([self.p_x, self.p_y])
        else:
            raise Exception('The format of the number of component is not supported.\n'
                            ' Please enter the number of components as a positive integer!')

        if self.n_comp > min([self.p_x, self.p_y]):
            raise Exception('The number of latent variables cannot be greater than the dimension of either matrices x '
                            'or y')

        if isinstance(tolerance, float):
            if tolerance > 1:
                raise Exception('Please enter a small value for tolerance. Ex. 1e-6')
            else:
                self.tolerance = tolerance
        elif not tolerance:
            self.tolerance = 1e-8
        else:
            raise Exception('The format of tolerance is not supported.\n'
                            ' Please enter a small value as tolerance (Ex. 1e-8)')

        if isinstance(max_iter, int):
            if max_iter < 1:
                raise Exception('Please enter a positive integer as the maximum number of iterations.')
            else:
                self.max_iter = max_iter
        elif not max_iter:
            self.max_iter = 1000
        else:
            raise Exception('The format of maximum iterations is not supported.\n'
                            ' Please enter positive integer as maximum number of iterations (Ex. 1000)')
        self.eps = jnp.finfo(float).eps
        self.itr = 0.0
        self.psi = random.uniform(key=self.key,
                                  shape=(self.p_z,),
                                  minval=0,
                                  maxval=1)

        self.f = random.uniform(key=self.key,
                                shape=(self.p_z, self.n_comp),
                                minval=0,
                                maxval=1)

        def _cond_fun(values: tuple = None) -> bool:
            itr, _, _, _, likelihood_error = values
            error = jnp.abs(likelihood_error).astype(float)
            return (error > self.tolerance) | (itr > self.max_iter)

        def _em_factor_analysis(values: tuple = None) -> tuple:
            itr, psi, f, old_log_likelihood, log_likelihood_error = values
            x_hat = jnp.diag(psi ** -0.5) @ self.z_m / jnp.sqrt(self.n_x)
            u_svd, s_svd, _ = jnp.linalg.svd(x_hat, full_matrices=False)
            a_svd = s_svd ** 2
            f = jnp.diag(psi ** 0.5) @ u_svd[:, :self.n_comp] @ jnp.diag(
                jnp.maximum(a_svd[:self.n_comp] - 1.0, self.eps) ** 0.5)
            likelihood = -0.5 * self.n_x * (jnp.log(a_svd[:self.n_comp]).sum() +
                                            self.n_comp + (a_svd[self.n_comp:]).sum() + jnp.log(
                        jnp.linalg.det(jnp.diag(psi * 2 * jnp.pi))))
            psi = self.var - jnp.diag(f @ f.T)
            log_likelihood_error = likelihood - old_log_likelihood
            itr += 1
            return itr, psi, f, likelihood, log_likelihood_error

        if self.method == 'EM':
            self.body_fun = _em_factor_analysis
            self.cond_fun = _cond_fun

    def fit(self):

        self.itr, \
            self.psi, self.f, \
            self.log_likelihood, \
            self.log_likelihood_error = lax.while_loop(body_fun=self.body_fun, cond_fun=self.cond_fun,
                                                       init_val=(self.itr,
                                                                 self.psi,
                                                                 self.f,
                                                                 jnp.array(-jnp.inf, dtype=jnp.float32),
                                                                 jnp.array(-jnp.inf, dtype=jnp.float32)))
        self.covariance_x = self.f[:self.p_x, :] @ self.f[:self.p_x, :].T + jnp.diag(self.psi[:self.p_x])
        self.covariance_y = self.f[self.p_x:, :] @ self.f[self.p_x:, :].T + jnp.diag(self.psi[self.p_x:])
        return self.covariance_x, self.covariance_y

    def fit_transform(self):
        self.fit()
        coef = self.f / self.psi[:, jnp.newaxis]
        return jnp.linalg.inv(coef.T @ self.f + jnp.eye(self.n_comp)) @ coef.T @ self.z_m

