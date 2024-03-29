import jax.numpy as jnp
from jax import lax, vmap, jit
import jax
from tqdm import tqdm


def ode_fcn_(x: jnp.ndarray = None, p: jnp.ndarray = None, t: jnp.ndarray = None, u: jnp.ndarray = None) -> jnp.ndarray:
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

d_dx = jax.vmap(fun=ode_fcn_, in_axes=[1, 1, None, None], out_axes=1)
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


# M = jax.lax.fori_loop(lower=0,
#                       upper=L,
#                       body_fun=wrapper,
#                       init_val=(x, par, u))


class ODESolvers:
    def __init__(self,
                 fcn: callable = None,
                 steps: int = None,
                 max_step_size: float = None,
                 duration: float = None,
                 n_sim: int = 1,
                 n_states: int = None,
                 n_params: int = None,
                 x0: jnp.ndarray = None,
                 method: str = 'euler',
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
            if method in ['euler', 'RK2', 'RK3', 'RK4', 'ralston', 'modified_euler', 'heun', 'AB2', 'AB3', 'AB4',
                          'AB5','ABAM1' ,'ABAM2', 'ABAM3', 'ABAM4', 'ABAM5']:
                self.method = method
            else:
                raise Exception('The specified method is not supported')
        elif not method:
            self.method = 'euler'
        else:
            raise Exception(f'Please enter the method for solving system of ordinary differential equation.')

        if self.method in ['AB2', 'AB3', 'AB4', 'AB5', 'ABAM2', 'ABAM3', 'ABAM4', 'ABAM5']:
            self.requires_init = True
        else:
            self.requires_init = False

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
            self.n_input = 1
        else:
            raise Exception('Please correctly specify whether the system of ode has input variable.')

        if isinstance(steps, int):
            self.steps = steps
        elif not steps:
            self.steps = None
        else:
            raise Exception('Please enter an integer to specify the number of iterations for simulation.')

        if isinstance(x0, jnp.ndarray):
            self.x0 = x0
        else:
            raise Exception('Please enter the initial condition of state variables.')

        # if self.x0.shape == (self.n_states, self.n_sim):
        #     pass

        if self.x0.shape == (self.n_states,):
            self.x0 = jnp.tile(A=self.x0[:, jnp.newaxis], reps=[1, self.n_sim])
        elif self.x0.shape == () or self.x0.shape == (1,):
            self.x0 = jnp.ones((self.n_states, self.n_sim)) * self.x0

        if not self.x0.shape == (self.n_states, self.n_sim):
            raise Exception('Given array of initial condition is not consistent with the number of state variables and'
                            'the number of simulation(parallel solution)')

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
        # reallocating input, parameters, and state variables
        self.x = jnp.zeros((self.n_states, self.n_sim, self.steps))
        self.x = self.x.at[:, :, 0].set(self.x0)
        self.parameters = jnp.ones((self.n_params, self.n_sim, self.steps))
        self.u = jnp.ones((self.n_input, self.n_sim, self.steps))
        # x: parallelized, p: parallelized, t: non-parallelized, u: non-parallelized
        if self.activate_jit:
            self.parallelized_odes = jit(jax.vmap(fun=self.fcn, in_axes=[1, 1, None, 1], out_axes=1,
                                                  axis_size=self.n_sim))
        else:
            self.parallelized_odes = jax.vmap(fun=self.fcn, in_axes=[1, 1, None, 1], out_axes=1,
                                              axis_size=self.n_sim)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if self.method == 'euler':
            self.lower_limit = 0
            self.upper_limit = self.steps - 1
            self.upper_limit_init = 0

            def fcn_main_euler(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                fn = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr])
                states = states.at[:, :, itr + 1].set(states[:, :, itr] + fn * self.delta[itr + 1])
                return states, parameters, inputs

            self.ode_parallel_wrapper = fcn_main_euler
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        elif self.method == 'RK2':
            self.lower_limit = 0
            self.upper_limit = self.steps - 1
            self.upper_limit_init = 0

            def fcn_main_rk2(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val

                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr]) \
                     * self.delta[itr]
                k2 = self.parallelized_odes(states[:, :, itr] + k1, parameters[:, :, itr], itr, inputs[:, :, itr]) * \
                     self.delta[itr]
                states = states.at[:, :, itr + 1].set(states[:, :, itr] + 0.5 * (k1 + k2))
                return states, parameters, inputs

            self.ode_parallel_wrapper = fcn_main_rk2
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        elif self.method == 'RK3':
            self.lower_limit = 0
            self.upper_limit = self.steps - 1
            self.upper_limit_init = 0

            def fcn_main_rk3(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr])
                k2 = self.parallelized_odes(states[:, :, itr] + 0.5 * self.delta[itr] * k1, parameters[:, :, itr],
                                            itr, inputs[:, :, itr])
                k3 = self.parallelized_odes(states[:, :, itr] - self.delta[itr] * k1 + 2 * self.delta[itr] * k2,
                                            parameters[:, :, itr], itr, inputs[:, :, itr])
                states = states.at[:, :, itr + 1].set(
                    states[:, :, itr] + (1 / 6) * self.delta[itr] * (k1 + 4 * k2 + k3))
                return states, parameters, inputs

            self.ode_parallel_wrapper = fcn_main_rk3

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        elif self.method == 'RK4':
            self.lower_limit = 0
            self.upper_limit = self.steps - 1
            self.upper_limit_init = 0

            def fcn_main_rk4(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val

                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr]) \
                     * self.delta[itr]
                k2 = self.parallelized_odes(states[:, :, itr] + 0.5 * k1, parameters[:, :, itr], itr,
                                            inputs[:, :, itr]) * \
                     self.delta[itr]
                k3 = self.parallelized_odes(states[:, :, itr] + 0.5 * k2, parameters[:, :, itr], itr,
                                            inputs[:, :, itr]) * \
                     self.delta[itr]
                k4 = self.parallelized_odes(states[:, :, itr] + k3, parameters[:, :, itr], itr, inputs[:, :, itr]) * \
                     self.delta[itr]
                states = states.at[:, :, itr + 1].set(states[:, :, itr] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4))
                return states, parameters, inputs

            self.ode_parallel_wrapper = fcn_main_rk4
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        elif self.method == 'ralston':
            self.lower_limit = 0
            self.upper_limit = self.steps - 1
            self.upper_limit_init = 0

            def fcn_main_ralston(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val

                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr])
                k2 = self.parallelized_odes(states[:, :, itr] + (2 / 3) * self.delta[itr] * k1, parameters[:, :, itr],
                                            itr, inputs[:, :, itr])
                states = states.at[:, :, itr + 1].set(states[:, :, itr] + 0.25 * self.delta[itr] * k1 +
                                                      0.75 * self.delta[itr] * k2)
                return states, parameters, inputs

            self.ode_parallel_wrapper = fcn_main_ralston
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        elif self.method == 'modified_euler':
            self.lower_limit = 0
            self.upper_limit = self.steps - 1
            self.upper_limit_init = 0

            def fcn_main_modified_euler(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val

                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr])
                k2 = self.parallelized_odes(states[:, :, itr] + self.delta[itr] * k1, parameters[:, :, itr],
                                            itr, inputs[:, :, itr])
                states = states.at[:, :, itr + 1].set(states[:, :, itr] + 0.5 * self.delta[itr] * (k2+k1))
                return states, parameters, inputs

            self.ode_parallel_wrapper = fcn_main_modified_euler
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        elif self.method == 'heun':
            self.lower_limit = 0
            self.upper_limit = self.steps - 1
            self.upper_limit_init = 0

            def fcn_main_heun(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr])
                k2 = self.parallelized_odes(states[:, :, itr] + (1 / 3) * self.delta[itr] * k1, parameters[:, :, itr],
                                            itr, inputs[:, :, itr])
                k3 = self.parallelized_odes(states[:, :, itr] + (2 / 3) * self.delta[itr] * k2, parameters[:, :, itr],
                                            itr, inputs[:, :, itr])
                states = states.at[:, :, itr + 1].set(states[:, :, itr] + 0.5 * self.delta[itr] * (k1 + k3))
                return states, parameters, inputs

            self.ode_parallel_wrapper = fcn_main_heun
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        elif self.method == 'AB3':
            self.lower_limit = 0
            self.upper_limit = self.steps - 3
            self.upper_limit_init = 2

            def fcn_main_ab3_init(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr])
                k2 = self.parallelized_odes(states[:, :, itr] + 0.5 * self.delta[itr] * k1, parameters[:, :, itr],
                                            itr, inputs[:, :, itr])
                k3 = self.parallelized_odes(states[:, :, itr] - self.delta[itr] * k1 + 2 * self.delta[itr] * k2,
                                            parameters[:, :, itr], itr, inputs[:, :, itr])
                states = states.at[:, :, itr + 1].set(
                    states[:, :, itr] + (1 / 6) * self.delta[itr] * (k1 + 4 * k2 + k3))
                return states, parameters, inputs

            def fcn_main_ab3(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                fn = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr])
                fn1 = self.parallelized_odes(states[:, :, itr + 1], parameters[:, :, itr + 1], itr + 1,
                                             inputs[:, :, itr + 1])
                fn2 = self.parallelized_odes(states[:, :, itr + 2], parameters[:, :, itr + 2], itr + 2,
                                             inputs[:, :, itr + 2])
                states = states.at[:, :, itr + 3].set(
                    states[:, :, itr + 2] + (self.delta[itr + 3] / 12) * (23 * fn2 - 16 * fn1 + 5 * fn))

                return states, parameters, inputs

            self.ode_parallel_wrapper = fcn_main_ab3
            self.ode_parallel_wrapper_init = fcn_main_ab3_init
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        elif self.method == 'AB2':
            self.lower_limit = 0
            self.upper_limit = self.steps - 2
            self.upper_limit_init = 1

            def fcn_main_ab2_init(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val

                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr]) \
                     * self.delta[itr]
                k2 = self.parallelized_odes(states[:, :, itr] + k1, parameters[:, :, itr], itr, inputs[:, :, itr]) * \
                     self.delta[itr]
                states = states.at[:, :, itr + 1].set(states[:, :, itr] + 0.5 * (k1 + k2))
                return states, parameters, inputs

            def fcn_main_ab2(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                fn = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr])
                fn1 = self.parallelized_odes(states[:, :, itr + 1], parameters[:, :, itr + 1], itr + 1,
                                             inputs[:, :, itr + 1])
                states = states.at[:, :, itr + 2].set(
                    states[:, :, itr + 1] + (self.delta[itr + 2] / 2) * (3 * fn1 - fn))
                return states, parameters, inputs

            self.ode_parallel_wrapper = fcn_main_ab2
            self.ode_parallel_wrapper_init = fcn_main_ab2_init
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        elif self.method == 'AB4':
            self.lower_limit = 0
            self.upper_limit = self.steps - 4
            self.upper_limit_init = 3

            def fcn_main_ab4_init(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val

                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr]) \
                     * self.delta[itr]
                k2 = self.parallelized_odes(states[:, :, itr] + 0.5 * k1, parameters[:, :, itr], itr,
                                            inputs[:, :, itr]) * self.delta[itr]
                k3 = self.parallelized_odes(states[:, :, itr] + 0.5 * k2, parameters[:, :, itr], itr,
                                            inputs[:, :, itr]) * self.delta[itr]
                k4 = self.parallelized_odes(states[:, :, itr] + k3, parameters[:, :, itr], itr, inputs[:, :, itr]) * \
                     self.delta[itr]
                states = states.at[:, :, itr + 1].set(states[:, :, itr] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4))
                return states, parameters, inputs

            def fcn_main_ab4(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                fn = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr])
                fn1 = self.parallelized_odes(states[:, :, itr + 1], parameters[:, :, itr + 1], itr + 1,
                                             inputs[:, :, itr + 1])
                fn2 = self.parallelized_odes(states[:, :, itr + 2], parameters[:, :, itr + 2], itr + 2,
                                             inputs[:, :, itr + 2])
                fn3 = self.parallelized_odes(states[:, :, itr + 3], parameters[:, :, itr + 3], itr + 3,
                                             inputs[:, :, itr + 3])

                states = states.at[:, :, itr + 4].set(
                    states[:, :, itr + 3] + (self.delta[itr + 4] / 24) * (55 * fn3 - 59 * fn2 + 37 * fn1 - 9 * fn))
                return states, parameters, inputs

            self.ode_parallel_wrapper = fcn_main_ab4
            self.ode_parallel_wrapper_init = fcn_main_ab4_init
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        elif self.method == 'AB5':
            self.lower_limit = 0
            self.upper_limit = self.steps - 5
            self.upper_limit_init = 4

            def fcn_main_ab5_init(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val

                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr]) \
                     * self.delta[itr]
                k2 = self.parallelized_odes(states[:, :, itr] + 0.5 * k1, parameters[:, :, itr], itr,
                                            inputs[:, :, itr]) * self.delta[itr]
                k3 = self.parallelized_odes(states[:, :, itr] + 0.5 * k2, parameters[:, :, itr], itr,
                                            inputs[:, :, itr]) * self.delta[itr]
                k4 = self.parallelized_odes(states[:, :, itr] + k3, parameters[:, :, itr], itr, inputs[:, :, itr]) * \
                     self.delta[itr]
                states = states.at[:, :, itr + 1].set(states[:, :, itr] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4))
                return states, parameters, inputs

            def fcn_main_ab5(itr, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                fn = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr])
                fn1 = self.parallelized_odes(states[:, :, itr + 1], parameters[:, :, itr + 1], itr + 1,
                                             inputs[:, :, itr + 1])
                fn2 = self.parallelized_odes(states[:, :, itr + 2], parameters[:, :, itr + 2], itr + 2,
                                             inputs[:, :, itr + 2])
                fn3 = self.parallelized_odes(states[:, :, itr + 3], parameters[:, :, itr + 3], itr + 3,
                                             inputs[:, :, itr + 3])
                fn4 = self.parallelized_odes(states[:, :, itr + 4], parameters[:, :, itr + 4], itr + 4,
                                             inputs[:, :, itr + 4])
                states = states.at[:, :, itr + 5].set(
                    states[:, :, itr + 4] + (self.delta[itr + 5] / 720) * (1901 * fn4 - 2774 * fn3 + 2616 * fn2
                                                                           - 1274 * fn1 + 251 * fn))
                return states, parameters, inputs

            self.ode_parallel_wrapper = fcn_main_ab5
            self.ode_parallel_wrapper_init = fcn_main_ab5_init
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        elif self.method == 'ABAM1':
            self.lower_limit = 0
            self.upper_limit = self.steps - 1
            self.upper_limit_init = 0

            def fcn_main_abam1(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                fn = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr])
                pn1 = states[:, :, itr] + self.delta[itr] * fn
                fp1 = self.parallelized_odes(pn1, parameters[:, :, itr + 1], itr + 1, inputs[:, :, itr + 1])
                states = states.at[:, :, itr + 1].set(states[:, :, itr] + self.delta[itr + 1] * fp1)
                return states, parameters, inputs

            self.ode_parallel_wrapper = fcn_main_abam1
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        elif self.method == 'ABAM2':
            self.lower_limit = 0
            self.upper_limit = self.steps - 2
            self.upper_limit_init = 1

            def fcn_main_abam2_init(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val

                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr]) \
                     * self.delta[itr]
                k2 = self.parallelized_odes(states[:, :, itr] + k1, parameters[:, :, itr], itr, inputs[:, :, itr]) * \
                     self.delta[itr]
                states = states.at[:, :, itr + 1].set(states[:, :, itr] + 0.5 * (k1 + k2))
                return states, parameters, inputs

            def fcn_main_abam2(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                fn = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr])
                fn1 = self.parallelized_odes(states[:, :, itr + 1], parameters[:, :, itr + 1], itr + 1,
                                             inputs[:, :, itr + 1])
                pn2 = states[:, :, itr + 1] + (self.delta[itr + 1] / 2) * (3 * fn1 - fn)
                fp2 = self.parallelized_odes(pn2, parameters[:, :, itr + 2], itr + 2, inputs[:, :, itr + 2])
                states = states.at[:, :, itr + 2].set(states[:, :, itr + 1] + 0.5 * self.delta[itr + 2] * (fn1 + fp2))
                return states, parameters, inputs

            self.ode_parallel_wrapper = fcn_main_abam2
            self.ode_parallel_wrapper_init = fcn_main_abam2_init
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        elif self.method == 'ABAM3':
            self.lower_limit = 0
            self.upper_limit = self.steps - 3
            self.upper_limit_init = 2

            def fcn_main_abam3_init(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr])
                k2 = self.parallelized_odes(states[:, :, itr] + 0.5 * self.delta[itr] * k1, parameters[:, :, itr],
                                            itr, inputs[:, :, itr])
                k3 = self.parallelized_odes(states[:, :, itr] - self.delta[itr] * k1 + 2 * self.delta[itr] * k2,
                                            parameters[:, :, itr], itr, inputs[:, :, itr])
                states = states.at[:, :, itr + 1].set(
                    states[:, :, itr] + (1 / 6) * self.delta[itr] * (k1 + 4 * k2 + k3))
                return states, parameters, inputs

            def fcn_main_abam3(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                fn = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr])
                fn1 = self.parallelized_odes(states[:, :, itr + 1], parameters[:, :, itr + 1], itr + 1,
                                             inputs[:, :, itr + 1])
                fn2 = self.parallelized_odes(states[:, :, itr + 2], parameters[:, :, itr + 2], itr + 2,
                                             inputs[:, :, itr + 2])
                pn3 = states[:, :, itr + 2] + (self.delta[itr + 2] / 12) * (23 * fn2 - 16 * fn1 + 5 * fn)
                fp3 = self.parallelized_odes(pn3, parameters[:, :, itr + 3], itr + 3, inputs[:, :, itr + 3])
                states = states.at[:, :, itr + 3].set(states[:, :, itr + 2] + (1 / 12) * self.delta[itr + 3]
                                                      * (5 * fp3 + 8 * fn2 - fn1))
                return states, parameters, inputs

            self.ode_parallel_wrapper = fcn_main_abam3
            self.ode_parallel_wrapper_init = fcn_main_abam3_init
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        elif self.method == 'ABAM4':

            self.lower_limit = 0
            self.upper_limit = self.steps - 3
            self.upper_limit_init = 3

            def fcn_main_abam4_init(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr]) \
                     * self.delta[itr]
                k2 = self.parallelized_odes(states[:, :, itr] + 0.5 * k1, parameters[:, :, itr], itr,
                                            inputs[:, :, itr]) * self.delta[itr]
                k3 = self.parallelized_odes(states[:, :, itr] + 0.5 * k2, parameters[:, :, itr], itr,
                                            inputs[:, :, itr]) * self.delta[itr]
                k4 = self.parallelized_odes(states[:, :, itr] + k3, parameters[:, :, itr], itr, inputs[:, :, itr]) * \
                     self.delta[itr]
                states = states.at[:, :, itr + 1].set(states[:, :, itr] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4))
                return states, parameters, inputs

            def fcn_main_abam4(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                fn = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr])
                fn1 = self.parallelized_odes(states[:, :, itr + 1], parameters[:, :, itr + 1], itr + 1,
                                             inputs[:, :, itr + 1])
                fn2 = self.parallelized_odes(states[:, :, itr + 2], parameters[:, :, itr + 2], itr + 2,
                                             inputs[:, :, itr + 2])
                fn3 = self.parallelized_odes(states[:, :, itr + 3], parameters[:, :, itr + 3], itr + 3,
                                             inputs[:, :, itr + 3])

                pn4 = states[:, :, itr + 3] + (self.delta[itr + 4] / 24) * (55 * fn3 - 59 * fn2 + 37 * fn1 - 9 * fn)
                fp4 = self.parallelized_odes(pn4, parameters[:, :, itr + 4], itr + 4, inputs[:, :, itr + 4])
                states = states.at[:, :, itr + 4].set(states[:, :, itr + 3] + (1 / 24) * self.delta[itr + 4]
                                                      * (9 * fp4 + 19 * fn3 - 5 * fn2 + fn1))
                return states, parameters, inputs

            self.ode_parallel_wrapper = fcn_main_abam4
            self.ode_parallel_wrapper_init = fcn_main_abam4_init

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        elif self.method == 'ABAM5':
            self.lower_limit = 0
            self.upper_limit = self.steps - 3
            self.upper_limit_init = 4

            def fcn_main_abam5_init(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                k1 = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr]) \
                     * self.delta[itr]
                k2 = self.parallelized_odes(states[:, :, itr] + 0.5 * k1, parameters[:, :, itr], itr,
                                            inputs[:, :, itr]) * self.delta[itr]
                k3 = self.parallelized_odes(states[:, :, itr] + 0.5 * k2, parameters[:, :, itr], itr,
                                            inputs[:, :, itr]) * self.delta[itr]
                k4 = self.parallelized_odes(states[:, :, itr] + k3, parameters[:, :, itr], itr, inputs[:, :, itr]) * \
                     self.delta[itr]
                states = states.at[:, :, itr + 1].set(states[:, :, itr] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4))
                return states, parameters, inputs

            def fcn_main_abam5(itr: int, init_val: tuple) -> tuple:
                states, parameters, inputs = init_val
                fn = self.parallelized_odes(states[:, :, itr], parameters[:, :, itr], itr, inputs[:, :, itr])
                fn1 = self.parallelized_odes(states[:, :, itr + 1], parameters[:, :, itr + 1], itr + 1,
                                             inputs[:, :, itr + 1])
                fn2 = self.parallelized_odes(states[:, :, itr + 2], parameters[:, :, itr + 2], itr + 2,
                                             inputs[:, :, itr + 2])
                fn3 = self.parallelized_odes(states[:, :, itr + 3], parameters[:, :, itr + 3], itr + 3,
                                             inputs[:, :, itr + 3])
                fn4 = self.parallelized_odes(states[:, :, itr + 4], parameters[:, :, itr + 4], itr + 4,
                                             inputs[:, :, itr + 4])
                pn5 = states[:, :, itr + 4] + (self.delta[itr + 5] / 720) * (1901 * fn4 - 2774 * fn3
                                                                             + 2616 * fn2 - 1274 * fn1 + 251 * fn)
                fp5 = self.parallelized_odes(pn5, parameters[:, :, itr + 5], itr + 5, inputs[:, :, itr + 5])
                states = states.at[:, :, itr + 5].set(states[:, :, itr + 4] + (1 / 720) * self.delta[itr + 5]
                                                      * (251 * fp5 + 646 * fn4 - 264 * fn3 + 106 * fn2 - 19 * fn1))
                return states, parameters, inputs

            self.ode_parallel_wrapper = fcn_main_abam5
            self.ode_parallel_wrapper_init = fcn_main_abam5_init

        def solve_with_init() -> jnp.ndarray:
            self.x, _, _ = lax.fori_loop(lower=self.lower_limit,
                                         upper=self.upper_limit_init,
                                         body_fun=self.ode_parallel_wrapper_init,
                                         init_val=(self.x, self.parameters, self.u))

            solution, _, _ = lax.fori_loop(lower=self.lower_limit,
                                           upper=self.upper_limit,
                                           body_fun=self.ode_parallel_wrapper,
                                           init_val=(self.x, self.parameters, self.u))
            return solution

        def solve_without_init() -> jnp.ndarray:
            solution, _, _ = lax.fori_loop(lower=self.lower_limit,
                                           upper=self.upper_limit,
                                           body_fun=self.ode_parallel_wrapper,
                                           init_val=(self.x, self.parameters, self.u))
            return solution

        self.solve_without_init = solve_without_init
        self.solve_with_init = solve_with_init
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def solve(self, parameter: jnp.ndarray = None, u: jnp.ndarray = None):

        if u == None:
            pass
        elif u.shape == (self.n_input, self.steps):
            self.u = jnp.tile(A=u[:, jnp.newaxis, :], reps=[1, self.n_sim, 1])
        elif u.shape == (self.n_input, self.n_sim, self.steps):
            self.u = u
        elif u.shape == (self.n_input,):
            self.u = jnp.tile(A=u[:, jnp.newaxis, jnp.newaxis], reps=[1, self.n_sim, self.steps])

        if parameter == None:
            pass
        elif parameter.shape == (self.n_params, self.steps):
            self.parameters = jnp.tile(A=u[:, jnp.newaxis, :], reps=[1, self.n_sim, 1])
        elif parameter.shape == (self.n_params, self.n_sim, self.steps):
            self.parameters = parameter
        elif parameter.shape == (self.n_params,):
            self.parameters = jnp.tile(A=u[:, jnp.newaxis, jnp.newaxis], reps=[1, self.n_sim, self.steps])

        if self.requires_init:
            return self.solve_with_init()
        else:
            return self.solve_without_init()


