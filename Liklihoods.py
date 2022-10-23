import numpy as np
import jax.numpy as jnp
from jax import lax, jit, vmap, grad

class Liklihood_Functions:
    def __int__(self, fcn: str = None, sigma: jnp.ndarray = None, covariance: jnp.ndarray = None,
                variant_chains: bool = False, activate_jit: bool = False, nchains: int = 1,
                estimated: np.ndarray = None, measured: np.ndarray = None, diagonal_covariance: bool = None,
                dimention: bool = None):

        # if not isinstance(vectorized, bool):
        #     self.vectorized = False
        #     print(
        #         f'------------------------------------------------------------------------------------------------------------------\n '
        #         f'The default value of {self.vectorized} is selectd for vectorizing simulations\n'
        #         f'---------------------------------------------------------------------------------------------------------------------')
        # else:
        #     self.vectorized = vectorized

        if not isinstance(nchains, int):
            self.nchains = 1
            print(
                f'--------------------------------------------------------------------------------------------------\n '
                f'The default value of {self.nchains} is selectd for the number of chains\n'
                f'----------------------------------------------------------------------------------------------------')
        else:
            self.nchains = nchains

        if not isinstance(dimention, int):
            raise Exception('The dimension of the problem is not specified correctly!')
        else:
            self.dimention = dimention

        # if not isinstance(N, int):
        #     raise Exception('The size of the data is not specified correctly!')
        # else:
        #     self.N = N

        if not isinstance(diagonal_covariance, bool):
            raise Exception('The type of calculation(vectorizing) is not specified correctly!')
        else:
            self.diagonal_covariance = diagonal_covariance

        if not isinstance(covariance, np.ndarray):
            raise Exception('The covariance matrix is not specified correctly!')
        else:
            self.covariance = covariance

        if not isinstance(sigma, (int, float)):
            raise Exception('The standard deviation is not specified correctly!')
        else:
            self.sigma = sigma

            if not isinstance(sigma, (int, float)):
                raise Exception('The standard deviation is not specified correctly!')
            else:
                self.sigma = sigma

        if not isinstance(measured, (int, np.ndarray)):
            raise Exception('the measured data is not entered correctly!')
        else:
            self.measured = measured

        if not isinstance(estimated, (int, np.ndarray)):
            raise Exception('the measured data is not entered correctly!')
        else:
            self.estimated = estimated

        if isinstance(fcn, str):
            self.fcn = fcn
        else:
            raise Exception('the type of likelihood function is not specified correctly!')
        #
        # if self.function is 'Normal' and not self.vectorized:
        #     self.liklihood = self.
        #
        # elif self.function is 'gaussian_single_variable' and self.vectorized:
        #
        #     self.liklihood = gaussian_liklihood_single_variable_vectorized
        #
        # elif self.function is 'multivariable_gaussian' and not self.vectorized:
        #     self.liklihood = gaussian_liklihood_multivariable
        #
        # elif self.function is 'multivariable_gaussian' and self.vectorized:
        #     self.liklihood = gaussian_liklihood_multivariable_vectorized
        #
        # else:
        #     raise Exception('The type of entered lklihood function is not implemented!')






 class Normal:
     def __init__(self):
         self.t = 1
     def liklihood(self, N, estimated: jnp.ndarray, measured: jnp.ndarray, sigma):
        error =  ((estimated - measured) ** 2).sum(axis=1)
        return ((sigma * jnp.sqrt(2 * jnp.pi))**(-N)) * jnp.exp((-0.5/sigma**2) * error)

     def diff_liklihood(self, N, estimated: jnp.ndarray, measured: jnp.ndarray, sigma):
         error = (estimated - measured)
         d_l_d_estimated = ((sigma * jnp.sqrt(2 * jnp.pi)) ** (-N)) * (-1/sigma**2) * (error.sum(axis=1)) *\
         jnp.exp((-0.5/sigma**2)*(error**2).sum(axis=1))

         d_l_d_sigma = jnp.exp((-0.5 / sigma ** 2) * error ** 2) * ((sigma * jnp.sqrt(2 * jnp.pi)) ** (-N)) *\
         ((-N*jnp.sqrt(2 * jnp.pi))/(sigma * jnp.sqrt(2 * jnp.pi)) + (1/sigma**3) * (error**2).sum(axis=1))

         return d_l_d_estimated, d_l_d_sigma
     def log_liklihood(self, N, estimated: jnp.ndarray, measured: jnp.ndarray, sigma):
         return -N * jnp.log(sigma * jnp.sqrt(2 * jnp.pi)) - (0.5/sigma**2) * ((estimated - measured) ** 2).sum(axis=1)
    def diff_log_liklihood(self, N, estimated: jnp.ndarray, measured: jnp.ndarray, sigma):
        dll_des = (-1 / sigma ** 2) * (estimated - measured).sum(axis=1)
        dll_sigma = (-N / sigma) + (1 / sigma**3) * ((estimated - measured)**2).sum(axis=1)
        return dll_des, dll_sigma



class MVNormal:
    def __init__(self, N: int = None, dim: int = None):
        self.dim = dim
        self.n = N
        self.diag_index = np.arange(self.dim, dtype=int)
        self.sample_index = np.arange(self.n, dtype=int)

    def liklihood(self, estimated: jnp.ndarray, measured: jnp.ndarray, covariance: jnp.ndarray):
        """
        #     The log liklihood of the Multivariable gaussian distribution used for multivariable fitting (multivariables objective function)
        #     :param measured: KxNxC measured parameters (K dimentional parameters and N sampling points and C chains)
        #     :param estimated:KxNxC estimated parameters (K dimentional parameters and N sampling points and C chains)
        #     :param N: An integer indicating the number of measurements
        #     :param Covariance: A positive definite square matrix indicating the covariance matrix of the multivariable Normal distribution (KxKxC)
        #     :param K: The dimention of the multivariable gaussian distribution
        #     :return: The log liklihood of the multivariable gaussian distribution
        #     """


        error = estimated - measured   # kxnxc
        error_t = jnp.transpose(error, axes=(1, 0, 2))  # nxkxc error matrix
        def det_inv(covariance):
            return jnp.linalg.det(covariance), jnp.linalg.inv(covariance)
        det,inversion = vmap(det_inv,in_axes=[2])(covariance)

        det,inversion = jnp.linalg.det(covariance), jnp.linalg.inv(covariance)

        def over_samples(xt, inv_cov, x):
            def over_chains(zt, inv_cov, z):
                return zt @ inv_cov @ z
            return vmap(over_chains, in_axes=[-1, None, -1])(xt, inv_cov, x)
        distance_points = vmap(over_samples, in_axes=[0, None, 1])(error_t, inversion, error)  # over samples
        mahalanobis_distance = distance_points.sum(axis=0)

        liklihood = ((2 * jnp.pi) ** (-0.5*self.dim*self.n)) * (det**(-0.5*self.n)) * jnp.exp(-0.5 * mahalanobis_distance)

        return liklihood
