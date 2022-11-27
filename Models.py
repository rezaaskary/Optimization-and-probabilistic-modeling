from tensorflow_probability.substrates.jax import distributions
from jax import vmap, jit, grad, random, lax, scipy, jacfwd
import jax.numpy as jnp
import sys
import warnings


class PPCA:
    def __init__(self, y: jnp.ndarray = None, n_comp: int = 2, max_iter: int = 500, tolerance: float = 1e-6):
        if not (min(y.shape) > 1):
            raise Exception('Too few feature variables')

        if not isinstance(y,jnp.ndarray):
            raise Exception('Invalid format of input matrix y!. Please enter the input matrix with ndarray format')
        else:
            self.y = y.T



def emppca_complete(y, k, w, v, maxiter, tolfun, tolx, dispnum, iterfmtstr):
    p, n = y.shape
    mu = jnp.mean(y, axis=1)[:, jnp.newaxis]
    y -= jnp.tile(mu, [1, n])
    iter = 0
    nloglk = jnp.inf
    traces = ((y.reshape((-1, 1))).T @ y.reshape((-1, 1))) / (n - 1)
    eps = jnp.finfo(float).eps
    while iter < maxiter:
        iter += 1
        sw = y @ (y.T @ w) / (n - 1)
        m = w.T @ w + v * jnp.eye(k)
        wnew = sw @ jnp.linalg.inv(v * jnp.eye(k) + jnp.linalg.inv(m) @ w.T @ sw)
        vnew = (traces - jnp.trace(sw @ jnp.linalg.inv(m) @ wnew.T)) / p

        dw = (jnp.abs(w - wnew) / (jnp.sqrt(eps) + (jnp.abs(wnew)).max())).max()
        dv = jnp.abs(v - vnew) / (eps + v)
        delta = max([dw, dv])
        cc = wnew @ wnew.T + vnew * jnp.eye(p)
        nloglk_new = (p * jnp.log(2 * jnp.pi) + jnp.log(jnp.linalg.det(cc)) + \
                      jnp.trace(jnp.linalg.inv(cc) @ y @ y.T / (n - 1))) * n / 2
        w = wnew
        v = vnew
        print(delta)
        print(jnp.abs(nloglk - nloglk_new))
        if delta < tolx:
            break
        elif (nloglk - nloglk_new) < tolfun:
            break
        elif jnp.abs(vnew) < jnp.sqrt(eps):
            break
        nloglk = nloglk_new
    ##=====================================================
    xmu = jnp.linalg.inv(m) @ wnew.T @ y
    return wnew, xmu, mu, vnew, iter, dw, nloglk_new


def PPCA(y: jnp.ndarray = None, k: int = 2):
    if not (min(y.shape) > 1): raise Exception('Too few feature variables')
    y = y.T
    nans = jnp.isnan(y)
    any_missing = jnp.any(nans)
    all_nan_rows = jnp.all(nans, axis=0)
    y = y[:, ~all_nan_rows]
    nans = nans[:, ~all_nan_rows]
    obs = ~nans
    num_obs = obs.sum()
    p, n = y.shape
    max_rank = min([n, p])
    flagWarnK = False
    if (not jnp.isscalar(k)) or (not isinstance(k, int)):
        raise Exception('invalid k!')
    elif k > max_rank - 1:
        k = max([1, maxRank - 1])
        flagWarnK = True
        print(
            f'Warning: Maximum possible rank of the data is {maxRank}. Computation continues with the number of principal components k set to {k}')
    else:
        pass

    param_names = ['Options', 'W0', 'v0']
    setflag = dict(zip(param_names, [0, 0, 0]))
    opt = [None]
    w = random.normal(key=random.PRNGKey(1), shape=(p, k))
    v = random.uniform(key=random.PRNGKey(1), shape=(1, 1))
    tolx = 1e-6
    tolfun = 1e-6
    maxiter = 2e4
    dispopt = 'off'
    if (setflag['W0']) and flagWarnK and w.shape[1] == max_rank:
        w = w[:, :-1]  # remove the last column
        print(f'Warning: Initial value W0 is truncated to {n} by {k}')

    if (setflag['W0']) and (jnp.any(jnp.isnan(w))):
        raise Exception(f'Initial matrix W0 must be a {p} by {k} numeric matrix without any NaN element')

    if (setflag['v0']) and (not (jnp.isscalar(k) and v > 0) or jnp.isnan(v) or n == jnp.inf):
        raise Exception('Initial residual variance v0 must be a positive scalar and must not be Inf.')

    if not sys.warnoptions:
        warnings.simplefilter('ignore')

    mu = jnp.zeros((p, 1))
    x = jnp.zeros((k, n))
    wnew = jnp.zeros((p, k))
    c = jnp.zeros((k, k, n))
    nloglk = jnp.inf

    dispnum = [1, 0, 0]
    headernames = ['Iteration      Variance     |Delta X|      Negative Log-likelihood']
    if dispnum[1]:
        print(headernames)

    itercount = 0

    if any_missing:
        while itercount < maxiter:
            itercount += 1
            for j in range(n):
                y_sample = y[:, j:j + 1]
                idxobs = obs[:, j]
                w_sample = w[idxobs, :]
                # Use Sherman-Morrison formula to find the inv(v.*eye(k)+w'*w)
                cj = jnp.eye(k) / v - (w_sample.T @ w_sample) @ jnp.linalg.inv(
                    jnp.eye(k) + (w_sample.T @ w_sample) / v) / (v ** 2)
                x = x.at[:, j:j + 1].set(cj @ (w_sample.T @ (y_sample[idxobs] - mu[idxobs])))
                c = c.at[:, :, j].set(cj)

            mu = jnp.nanmean(y - w @ x, axis=1)[:, jnp.newaxis]
            for i in range(p):
                idxobs = obs[i, :]
                m = x[:, idxobs] @ x[:, idxobs].T + v * jnp.sum(c[:, :, idxobs], axis=2)
                ww = x[:, idxobs] @ (y[i, idxobs] - mu[i, 0]).T
                # wnew[i, :] = jnp.linalg.solve(m, ww)
                wnew = wnew.at[i, :].set(jnp.linalg.solve(m, ww))
            vsum = jnp.zeros((1, 1))
            for j in range(n):
                wnew_sample = wnew[obs[:, j], :]
                vsum = vsum + ((y[obs[:, j], j] - wnew_sample @ x[:, j] - mu[obs[:, j], 0]) ** 2 + \
                               v * (jnp.diag(wnew_sample @ c[:, :, j] @ wnew_sample.T))).sum()

            vnew = vsum / num_obs
            eps = jnp.finfo(float).eps
            nloglk_new = 0

            for j in range(n):
                idxobs = obs[:, j]
                y_c = y[idxobs, j:j + 1] - mu[obs[:, j], 0:1]

                wobs = wnew[idxobs, :]
                cy = wobs @ wobs.T + vnew * jnp.eye(sum(idxobs))
                nloglk_new = nloglk_new + (sum(idxobs) * jnp.log(2 * jnp.pi) + jnp.log(jnp.linalg.det(cy)) +
                                           jnp.trace(jnp.linalg.inv(cy) @ y_c @ y_c.T)) / 2

            dw = (jnp.abs(w - wnew) / (jnp.sqrt(eps) + (jnp.abs(wnew)).max())).max()

            w = wnew
            v = vnew
            print(jnp.abs(nloglk - nloglk_new))
            if jnp.abs(nloglk - nloglk_new) < tolfun:
                break
            nloglk = nloglk_new
        mux = jnp.mean(x, axis=1)[:, jnp.newaxis]
        x = x - jnp.tile(mux, [1, n])
        mu = mu + w @ mux

    else:
        iterfmtstr = ''
        w, x, mu, v, itercount, dw, nloglk = emppca_complete(y, k, w, v, maxiter, tolfun, tolx, dispnum, iterfmtstr)

    if jnp.all(w == 0):
        coeff = jnp.zeros((p, k))
        coeff[::(p + 1)] = 1
        score = jnp.zeros((n, k))
        latent = jnp.zeros((k, 1))
        mu = (jnp.mean(y, axis=1)).T
        v = 0
        rsltStruct = {'W': w, \
                      'Xexp': x.T, \
                      'Recon': jnp.tile(mu, [n, 1]), \
                      'v': v, \
                      'NumIter': itercount, \
                      'RMSResid': 0, \
                      'nloglk': nloglk
                      }
        return coeff, score, latent, mu, v, rsltStruct


if __name__ == '__main__':
    data = random.gamma(key=random.PRNGKey(23), a=0.2, shape=(50, 5))
    data = data.at[2, 4].set(jnp.nan)
    PPCA(y=data, k=2)
