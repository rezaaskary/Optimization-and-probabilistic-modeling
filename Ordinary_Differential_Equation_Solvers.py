import jax.numpy as jnp
from jax import lax, vmap


class ODESolvers:
    def __int__(self,
                fcn: callable = None,
                n: int = None,
                method: str = 'Euler'):
        if isinstance(method, str):
            if method not in ['Euler', 'rk2', 'rk4']:
                self.method = method
            else:
                raise Exception('The specified method is not supported')

