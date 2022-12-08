import jax.numpy as jnp
from jax import lax, vmap
import jax

def ode_fcn(x: jnp.ndarray = None, p: jnp.ndarray = None, t: jnp.ndarray = None) -> jnp.ndarray:
    m = 4  # the number of state variables
    dx_dt = jnp.zeros((m, 1))  # reallocating the values of state variables
    dx0_dt = -6*x[0] + x[1]*x[2] * p[0]
    dx1_dt = -2 * x[1] + jnp.sin(x[3] + x[2])
    dx2_dt = -8 * x[2] + jnp.sin(x[0] + x[1] * p[2])
    dx3_dt = -3 * x[3] + jnp.cos(x[3] ** p[3] * x[1])

    dx_dt = dx_dt.at[0, 0].set(dx0_dt)
    dx_dt = dx_dt.at[1, 0].set(dx1_dt)
    dx_dt = dx_dt.at[2, 0].set(dx2_dt)
    dx_dt = dx_dt.at[3, 0].set(dx3_dt)
    return dx_dt

n_par = 3
chains = 1000
x = jax.random.uniform(key=jax.random.PRNGKey(7), minval=-4, maxval=4, shape=(n_par, chains), dtype=jnp.float64)














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
        elif not method:
            self.method = 'Euler'
        else:
            raise Exception(f'Please enter the method for solving system of ordinary differential equation.')
