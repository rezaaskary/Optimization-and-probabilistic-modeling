import jax.numpy as jnp
from jax import lax, vmap
import jax
from tqdm import tqdm


def ode_fcn(x: jnp.ndarray = None, p: jnp.ndarray = None, t: jnp.ndarray = None, u: jnp.ndarray = None) -> jnp.ndarray:
    m = 4  # the number of state variables
    dx_dt = jnp.zeros((m, 1))  # reallocating the values of state variables
    dx0_dt = -6 * x[0] + x[1] * x[2] * p[0] + u[0]
    dx1_dt = -2 * x[1] + jnp.sin(x[3] + x[2])
    dx2_dt = -8 * x[2] + jnp.sin(x[0] + x[1] * p[2]) + u[1]
    dx3_dt = -3 * x[3] + jnp.cos(x[3] ** p[3] * x[1])

    dx_dt = dx_dt.at[0, 0].set(dx0_dt)
    dx_dt = dx_dt.at[1, 0].set(dx1_dt)
    dx_dt = dx_dt.at[2, 0].set(dx2_dt)
    dx_dt = dx_dt.at[3, 0].set(dx3_dt)

    return dx_dt


n_par = 3
chains = 10000
L = 10000
par = jax.random.uniform(key=jax.random.PRNGKey(7), minval=-4, maxval=4, shape=(n_par, chains), dtype=jnp.float64)
x_0 = jax.random.uniform(key=jax.random.PRNGKey(7), minval=-4, maxval=4, shape=(4, chains), dtype=jnp.float64)
u = jax.random.uniform(key=jax.random.PRNGKey(7), minval=-4, maxval=4, shape=(2, L), dtype=jnp.float64)

d_dx = jax.vmap(fun=ode_fcn, in_axes=[1, 1, None, None], out_axes=1)
x = jnp.zeros((4, chains, L))
x = x.at[:, :, 0].set(x_0)
delta = 1


# for i in tqdm(range(L)):


def wrapper(itr: int, init_val: tuple) -> tuple:
    x, par, u = init_val
    evaluation = d_dx(x[:, :, itr], par, itr, u[:, itr])[:, :, 0]
    # [:, :, 0]
    x=x.at[:, :, itr + 1].set(evaluation)
    return x, par, u


M = jax.lax.fori_loop(lower=0,
                      upper=L,
                      body_fun=wrapper,
                      init_val=(x, par, u))
M


class ODESolvers:
    def __int__(self,
                fcn: callable = None,
                iteration: int = None,
                delta: float = None,
                duration: float = None,
                n_sim: int = 1,
                n_states: int = None,
                method: str = 'Euler',
                activate_jit: bool = False,
                has_input: bool = False):

        if isinstance(method, str):
            if method not in ['Euler', 'rk2', 'rk4']:
                self.method = method
            else:
                raise Exception('The specified method is not supported')
        elif not method:
            self.method = 'Euler'
        else:
            raise Exception(f'Please enter the method for solving system of ordinary differential equation.')

        if isinstance(n_sim, int):
            self.n_sim = n_sim
        elif not n_sim:
            self.n_sim = 1
        else:
            raise Exception('The number of parallel simulation is not specified correctly.')

        if isinstance(activate_jit, bool):
            self.activate_jit = activate_jit
        elif not activate_jit:
            self.activate_jit = False
        else:
            raise Exception('Please correctly specify jit-in-time compilation feature.')

        if isinstance(has_input, bool):
            self.has_input = has_input
        else:
            raise Exception('Please correctly specify whether the system of ode has input variable.')

        if isinstance(iteration, int):
            self.iteration = iteration
        elif not iteration:
            self.iteration = None
        else:
            raise Exception('Please enter an integer to specify the number of iterations for simulation.')


        if isinstance(duration, int):
            self.duration = duration
        elif not duration:
            self.duration = None
        else:
            raise Exception('Please enter a float value to specify the duration of simulation.')






        if iteration and duration and not delta:
        elif iteration and delta and not duration:
            elif delta and duration and not




            if







        else:
            raise Exception('Please correctly specify the number of iteration for solving the system of ode.')














