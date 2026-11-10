import jax
import jax.numpy as jnp
from jax import lax, random, jit
import diffrax

from flow.utils import scan_with_init_state, scan_wrapper
from functools import partial

class Integrator():
    """Base class for fixed-grid ODE/SDE integrators.

    Subclasses implement ``step`` (a single update); ``solve`` returns the final
    state after ``n_steps`` and ``trajectory`` returns the full path. The vector
    field ``ode`` is called as ``ode(x, t, *args)`` and ``diffusion`` (SDEs
    only) as ``diffusion(t)``.
    """
    def __init__(self, ode, diffusion=None, **kwargs):
        self.ode = ode
        self.diffusion = diffusion
        self.kwargs = kwargs

    def step(self, x, *args, **kwargs):
        raise NotImplementedError

    def solve(self, x_0, t0, t1, n_steps, key, *args):
        ts = jnp.linspace(t0, t1, n_steps + 1)
        dt = ts[1] - ts[0]
        if key is not None:
            keys = random.split(key, n_steps)
        else:
            keys = jnp.zeros((n_steps, 2), dtype=jnp.uint32)

        def step_fn(i, x):
            return self.step(x, ts[i], dt, keys[i], *args)

        return lax.fori_loop(0, n_steps, step_fn, x_0)

    def trajectory(self, x_0, t0, t1, n_steps, key, *args):
        ts = jnp.linspace(t0, t1, n_steps + 1)
        dt = ts[1] - ts[0]
        if key is not None:
            keys = random.split(key, n_steps)
        else:
            keys = jnp.zeros((n_steps, 2), dtype=jnp.uint32)

        def step_fn(x, i):
            return self.step(x, ts[i], dt, keys[i], *args)

        def scan_fn(acc, i):
            y = step_fn(acc, i)
            return y, y

        return scan_with_init_state(scan_fn, x_0, jnp.arange(n_steps + 1), length=n_steps + 1)

def get_integrator(name):
    """Return the integrator class (or partial) for a string name.

    Recognised names: ``'euler'``, ``'rk4'``, ``'euler-maruyama'``, and
    ``'diffrax'`` (Dopri5 by default; ``'diffrax-kvaerno5'`` for a stiff solver).
    """
    if name == 'euler':
        return Euler
    elif name == 'rk4':
        return RK4
    elif name == 'euler-maruyama':
        return EulerMaruyama
    elif 'diffrax' in name:
        if 'dopri5' in name or 'diffrax' == name:  # default to dopri5
            solver = diffrax.Dopri5()
        elif 'kvaerno5' in name:
            solver = diffrax.Kvaerno5()
        else:
            raise ValueError(f'Unknown integrator {name}')
        return partial(DiffraxIntegrator, solver=solver)
    else:
        raise ValueError(f'Unknown integrator {name}')

class DiffraxIntegrator(Integrator):
    """Adaptive ODE integration via `diffrax` (Dopri5 with a PID step controller
    by default). ``n_steps`` only sets the initial step size ``dt0``."""
    def __init__(self, ode, diffusion=None, solver=None, stepsize_controller=None, **kwargs):
        super().__init__(ode, diffusion, **kwargs)
        self.solver = solver if solver is not None else diffrax.Dopri5()
        # self.solver = solver if solver is not None else diffrax.Kvaerno5()
        self.stepsize_controller = stepsize_controller if stepsize_controller is not None else diffrax.PIDController(rtol=1e-5, atol=1e-5)

    @property
    def _solve_kwargs(self):
        """Extra diffeqsolve options; dt0 is passed explicitly, so it is not repeated here."""
        return {k: v for k, v in self.kwargs.items() if k != 'dt0'}

    def solve(self, x_0, t0, t1, n_steps, key, *args):        
        # Diffrax expects (t, y, args)
        def diffrax_ode(t, y, args_):
            return self.ode(y, t, *args_)

        term = diffrax.ODETerm(diffrax_ode)
        dt0 = self.kwargs.get('dt0', (t1 - t0) / n_steps if n_steps > 0 else None)

        sol = diffrax.diffeqsolve(
            term,
            self.solver,
            t0,
            t1,
            dt0=dt0,
            y0=x_0,
            args=args,
            stepsize_controller=self.stepsize_controller,
            saveat=diffrax.SaveAt(t1=True),
            **self._solve_kwargs
        )
        return jax.tree_util.tree_map(lambda y: y[-1], sol.ys)

    def solve_with_stats(self, x_0, t0, t1, n_steps, key, *args):
        """Like solve(), but also returns the adaptive solver's statistics
        (dict with 'num_steps', 'num_accepted_steps', 'num_rejected_steps').
        The step count at fixed tolerance is a direct stiffness measure."""
        def diffrax_ode(t, y, args_):
            return self.ode(y, t, *args_)

        term = diffrax.ODETerm(diffrax_ode)
        dt0 = self.kwargs.get('dt0', (t1 - t0) / n_steps if n_steps > 0 else None)
        sol = diffrax.diffeqsolve(
            term, self.solver, t0, t1, dt0=dt0, y0=x_0, args=args,
            stepsize_controller=self.stepsize_controller,
            saveat=diffrax.SaveAt(t1=True), **self._solve_kwargs
        )
        y1 = jax.tree_util.tree_map(lambda y: y[-1], sol.ys)
        stats = {k: v for k, v in sol.stats.items() if 'step' in k}
        return y1, stats

    def trajectory(self, x_0, t0, t1, n_steps, key, *args):

        def diffrax_ode(t, y, args_):
            return self.ode(y, t, *args_)

        ts = jnp.linspace(t0, t1, n_steps + 1)
        term = diffrax.ODETerm(diffrax_ode)
        dt0 = self.kwargs.get('dt0', (t1 - t0) / n_steps if n_steps > 0 else None)

        sol = diffrax.diffeqsolve(
            term,
            self.solver,
            t0,
            t1,
            dt0=dt0,
            y0=x_0,
            args=args,
            stepsize_controller=self.stepsize_controller,
            saveat=diffrax.SaveAt(ts=ts),
            **self._solve_kwargs
        )
        return sol.ys

class Euler(Integrator):
    """Explicit (forward) Euler ODE integrator."""
    def step(self, x, t, dt, unused_key, *args):
        # x + ode(x, t) * dt
        dxdt = self.ode(x, t, *args)
        return jax.tree_util.tree_map(lambda x_i, v_i: x_i + v_i * dt, x, dxdt)

class RK4(Integrator):
    """Classic 4th-order Runge-Kutta ODE integrator."""
    def step(self, x, t, dt, unused_key, *args):
        # k1 = f(x, t)
        k1 = self.ode(x, t, *args)
        
        # k2 = f(x + 0.5*dt*k1, t + 0.5*dt)
        x_k2 = jax.tree_util.tree_map(lambda x_i, k1_i: x_i + 0.5 * dt * k1_i, x, k1)
        k2 = self.ode(x_k2, t + 0.5 * dt, *args)
        
        # k3 = f(x + 0.5*dt*k2, t + 0.5*dt)
        x_k3 = jax.tree_util.tree_map(lambda x_i, k2_i: x_i + 0.5 * dt * k2_i, x, k2)
        k3 = self.ode(x_k3, t + 0.5 * dt, *args)
        
        # k4 = f(x + dt*k3, t + dt)
        x_k4 = jax.tree_util.tree_map(lambda x_i, k3_i: x_i + dt * k3_i, x, k3)
        k4 = self.ode(x_k4, t + dt, *args)
        
        # x_new = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        def update(x_i, k1_i, k2_i, k3_i, k4_i):
            return x_i + (dt / 6.0) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)
            
        return jax.tree_util.tree_map(update, x, k1, k2, k3, k4)

class EulerMaruyama(Integrator):
    """Euler-Maruyama SDE integrator: ``x + drift*dt + diffusion*sqrt(dt)*noise``.

    Requires a ``diffusion`` function. Noise is generated per PyTree leaf with a
    freshly split key so that structured (dict) states are handled correctly.
    """
    def __init__(self, ode, diffusion, **kwargs):
        if diffusion is None:
            raise ValueError("EulerMaruyama integrator requires a diffusion function.")
        super().__init__(ode, diffusion, **kwargs)

    def step(self, x, t, dt, key, *args):
        # x + drift*dt + diffusion*sqrt(dt)*noise
        drift = self.ode(x, t, *args)
        diffusion_term = self.diffusion(t)

        # Generate independent noise for each PyTree leaf (one split key per leaf,
        # so reused keys never produce correlated noise across leaves).
        leaves, treedef = jax.tree_util.tree_flatten(x)
        keys = random.split(key, len(leaves))
        flat_noise = [random.normal(keys[i], leaf.shape) for i, leaf in enumerate(leaves)]
        noise = jax.tree_util.tree_unflatten(treedef, flat_noise)

        def update(x_i, d_i, n_i):
            return x_i + d_i * dt + diffusion_term * jnp.sqrt(dt) * n_i

        return jax.tree_util.tree_map(update, x, drift, noise)