from tensorflow_probability.substrates.jax import distributions
from jax import vmap, jit, grad, random, lax, scipy, jacfwd
import jax.numpy as jnp




class PPCA:
    def __init__(self,y: jnp.ndarray = None, k: int = 2):
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
















if __name__=='__main__':
    data = random.gamma(key=random.PRNGKey(23),a=0.2,shape=(50,5))

    PPCA(y=data,k=2)
