from tensorflow_probability.substrates.jax import distributions
from jax import vmap, jit, grad, random, lax, scipy, jacfwd
import jax.numpy as jnp




class PPCA:
    def __init__(self,Y: jnp.ndarray = None, ):






def emppca_complete(Y, k, W, v, MaxIter, TolFun, TolX, dispnum, iterfmtstr):
    p, n = Y.shape
    mu = np.mean(Y, axis=1)[:, np.newaxis]
    Y -= np.tile(mu, [1, n])
    iter = 0
    nloglk = np.inf
    traceS = ((Y.reshape((-1, 1))).T @ Y.reshape((-1, 1))) / (n - 1)
    eps = np.finfo(float).eps
    while iter < MaxIter:
        iter += 1
        SW = Y @ (Y.T @ W) / (n - 1)
        M = W.T @ W + v * np.eye(k)
        Wnew = SW @ np.linalg.inv(v * np.eye(k) + np.linalg.inv(M) @ W.T @ SW)
        vnew = (traceS - np.trace(SW @ np.linalg.inv(M) @ Wnew.T)) / p

        dw = (np.abs(W - Wnew) / (np.sqrt(eps) + (np.abs(Wnew)).max())).max()
        dv = np.abs(v - vnew) / (eps + v)
        delta = max([dw, dv])
        CC = Wnew @ Wnew.T + vnew * np.eye(p)
        nloglk_new = (p * np.log(2 * np.pi) + np.log(np.linalg.det(CC)) + \
                      np.trace(np.linalg.inv(CC) @ Y @ Y.T / (n - 1))) * n / 2
        W = Wnew
        v = vnew
        print(delta)
        print(np.abs(nloglk - nloglk_new))
        if delta < TolX:
            break
        elif (nloglk - nloglk_new) < TolFun:
            break
        elif np.abs(vnew) < np.sqrt(eps):
            break
        nloglk = nloglk_new
    ##=====================================================
    Xmu = np.linalg.inv(M) @ Wnew.T @ Y
    return Wnew, Xmu, mu, vnew, iter, dw, nloglk_new