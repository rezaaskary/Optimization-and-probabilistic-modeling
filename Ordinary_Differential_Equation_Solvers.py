import jax.numpy as jnp
from jax import lax, vmap, jit
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


D = len(ode_fcn)

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
    x = x.at[:, :, itr + 1].set(evaluation)
    return x, par, u


M = jax.lax.fori_loop(lower=0,
                      upper=L,
                      body_fun=wrapper,
                      init_val=(x, par, u))
M


class ODESolvers:
    def __int__(self,
                fcn: callable = None,
                steps: int = None,
                max_step_size: float = None,
                duration: float = None,
                n_sim: int = 1,
                n_states: int = None,
                n_params: int = None,
                method: str = 'Euler',
                activate_jit: bool = False,
                n_input: int = None):
        """
        reference:
        <<Süli, Endre. "Numerical solution of ordinary differential equations." Mathematical Institute,
         University of Oxford (2010).>>
        :param fcn:
        :param steps:
        :param max_step_size:
        :param duration:
        :param n_sim:
        :param n_states:
        :param n_params:
        :param method:
        :param activate_jit:
        :param n_input:
        :return:
        """

        if hasattr(fcn, "__call__"):
            self.fcn = fcn
        else:
            raise Exception('The function of ode is not specified properly!')

        if isinstance(method, str):
            if method not in ['euler', 'RK2', 'RK4', 'ralston', 'modified_euler', 'heun', 'RK3']:
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

        if isinstance(n_params, int):
            self.n_params = n_params
        elif not n_params:
            self.n_params = None
        else:
            raise Exception('Please specify the number of parameters in the system of odes')

        if isinstance(n_states, int):
            self.n_states = n_states
        else:
            raise Exception('The number of ODEs is not specified correctly.')

        if isinstance(activate_jit, bool):
            self.activate_jit = activate_jit
        elif not activate_jit:
            self.activate_jit = False
        else:
            raise Exception('Please correctly specify jit-in-time compilation feature.')

        if isinstance(n_input, int):
            self.n_input = n_input
        elif not n_input:
            self.n_input = None
        else:
            raise Exception('Please correctly specify whether the system of ode has input variable.')

        if isinstance(steps, int):
            self.steps = steps
        elif not steps:
            self.steps = None
        else:
            raise Exception('Please enter an integer to specify the number of iterations for simulation.')

        if isinstance(duration, (int, float)):
            self.duration = duration
        elif not duration:
            self.duration = None
        else:
            raise Exception('Please enter a float value to specify the duration of simulation.')

        if isinstance(max_step_size, (int, float)):
            self.max_step_size = max_step_size
        elif not max_step_size:
            self.max_step_size = None
        else:
            raise Exception('Please enter a positive value to specify the length of interval for solving the system of'
                            ' ODE.')

        if self.steps and self.duration and not self.max_step_size:
            self.max_step_size = self.duration / self.steps

        elif self.steps and self.max_step_size and not self.duration:
            self.duration = self.steps * self.max_step_size

        elif self.max_step_size and self.duration and not self.steps:
            self.steps = self.duration // self.max_step_size + 1
            self.delta = jnp.ones((self.steps,)) * self.max_step_size
            self.delta = self.delta.at[-1].set(self.duration % self.max_step_size)

        elif self.steps and not self.duration and not self.max_step_size:
            raise Exception('Please enter either the duration of simulation or max_step_size of simulation')

        elif not self.steps and self.max_step_size and not self.duration:
            raise Exception('Please enter either the duration of simulation or steps of simulation')

        elif not self.steps and not self.max_step_size and self.duration:
            raise Exception('Please enter either the max_step_size or steps of simulation')

        elif self.steps and self.duration and self.max_step_size:
            raise Exception('Over determination! only two values for solving ODEs are required.')

        else:
            raise Exception('Please enter values for two input variables "steps", "max_step_size", or "duration" to'
                            ' solve equations.')

        if self.max_step_size <= 0:
            raise Exception('The length of steps must be a positive value.')

        if self.steps <= 0:
            raise Exception('The number of iterations must be a positive value.')

        if self.duration <= 0:
            raise Exception('The duration of simulation must be a positive value.')

        if not 'self.delta' in locals():
            self.delta = jnp.ones((self.steps,)) * self.max_step_size
        # checking the input arguments
        if not self.fcn.__code__.co_argcount == 4:
            raise Exception(f' The number of the input arguments of the function of odes should be 4. '
                            f' with the order of "x": the array of state variables, "p": the array of parameters,'
                            f' "t": the index of the step(counter), and "u": the array of the exogenous inputs of'
                            f' the system of odes at step i')

        self.x = jnp.zeros((self.n_states, self.n_sim, self.steps))
        self.parameters = jnp.ones((self.n_params, self.n_sim, self.steps))

        # x: parallelized, p: parallelized, t: non-parallelized, u: non-parallelized
        if self.activate_jit:
            self.parallelized_odes = jit(jax.vmap(fun=ode_fcn, in_axes=[1, 1, None, 1], out_axes=1))
        else:
            self.parallelized_odes = jax.vmap(fun=ode_fcn, in_axes=[1, 1, None, 1], out_axes=1)

        if self.method == 'euler':
            def ode_parallel_wrapper(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                evaluation = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, u[:, :, itr])
                states = states.at[:, :, itr + 1].set(states[:, :, itr] + evaluation * self.delta[itr])
                return states, parameters, inputs
        elif self.method == 'RK2':
            def ode_parallel_wrapper(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val

                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, u[:, :, itr]) \
                     * self.delta[itr]
                k2 = self.parallelized_odes(states[:, :, itr] + k1, parameters[:, :, itr], itr, u[:, :, itr]) * \
                     self.delta[itr]
                states = states.at[:, :, itr + 1].set(states[:, :, itr] + 0.5 * (k1 + k2))
                return states, parameters, inputs

        elif self.method == 'RK4':
            def ode_parallel_wrapper(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val

                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, u[:, :, itr]) \
                     * self.delta[itr]
                k2 = self.parallelized_odes(states[:, :, itr] + 0.5 * k1, parameters[:, :, itr], itr, u[:, :, itr]) * \
                     self.delta[itr]
                k3 = self.parallelized_odes(states[:, :, itr] + 0.5 * k2, parameters[:, :, itr], itr, u[:, :, itr]) * \
                     self.delta[itr]
                k4 = self.parallelized_odes(states[:, :, itr] + k3, parameters[:, :, itr], itr, u[:, :, itr]) * \
                     self.delta[itr]
                states = states.at[:, :, itr + 1].set(states[:, :, itr] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4))
                return states, parameters, inputs

        elif self.method == 'ralston':
            def ode_parallel_wrapper(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val

                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, u[:, :, itr])
                k2 = self.parallelized_odes(states[:, :, itr] + (2 / 3) * self.delta[itr] * k1, parameters[:, :, itr],
                                            itr, u[:, :, itr])
                states = states.at[:, :, itr + 1].set(states[:, :, itr] + 0.25 * self.delta[itr] * k1 +
                                                      0.75 * self.delta[itr] * k2)
                return states, parameters, inputs

        elif self.method == 'modified_euler':
            def ode_parallel_wrapper(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val

                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, u[:, :, itr])
                k2 = self.parallelized_odes(states[:, :, itr] + 0.5 * self.delta[itr] * k1, parameters[:, :, itr],
                                            itr, u[:, :, itr])
                states = states.at[:, :, itr + 1].set(states[:, :, itr] + 0.5 * self.delta[itr] * k2)
                return states, parameters, inputs

        elif self.method == 'heun':

            def ode_parallel_wrapper(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, u[:, :, itr])
                k2 = self.parallelized_odes(states[:, :, itr] + (1 / 3) * self.delta[itr] * k1, parameters[:, :, itr],
                                            itr, u[:, :, itr])
                k3 = self.parallelized_odes(states[:, :, itr] + (2 / 3) * self.delta[itr] * k2, parameters[:, :, itr],
                                            itr, u[:, :, itr])
                states = states.at[:, :, itr + 1].set(states[:, :, itr] + 0.25 * self.delta[itr] * (k1 + k3))
                return states, parameters, inputs


        elif self.method == 'RK3':
            def ode_parallel_wrapper(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, u[:, :, itr])
                k2 = self.parallelized_odes(states[:, :, itr] + 0.5 * self.delta[itr] * k1, parameters[:, :, itr],
                                            itr, u[:, :, itr])
                k3 = self.parallelized_odes(states[:, :, itr] - self.delta[itr] * k1 + 2 * self.delta[itr] * k2,
                                            parameters[:, :, itr], itr, u[:, :, itr])
                states = states.at[:, :, itr + 1].set(
                    states[:, :, itr] + (1 / 6) * self.delta[itr] * (k1 + 4 * k2 + k3))
                return states, parameters, inputs

        elif self.method == 'AB_2nd':
            def ode_parallel_wrapper(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, u[:, :, itr])
                k2 = self.parallelized_odes(states[:, :, itr] + 0.5 * self.delta[itr] * k1, parameters[:, :, itr],
                                            itr, u[:, :, itr])
                k3 = self.parallelized_odes(states[:, :, itr] - self.delta[itr] * k1 + 2 * self.delta[itr] * k2,
                                            parameters[:, :, itr], itr, u[:, :, itr])
                states = states.at[:, :, itr + 1].set(
                    states[:, :, itr] + (1 / 6) * self.delta[itr] * (k1 + 4 * k2 + k3))
                return states, parameters, inputs




        self.ode_parallel_wrapper = ode_parallel_wrapper
