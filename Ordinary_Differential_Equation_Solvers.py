import jax.numpy as jnp
from jax import lax, vmap


class ODESolvers:
    def __int__(self,
                fcn: callable = None,
                n: int = None,
                method: str = 'Euler'):
        return
