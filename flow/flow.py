import jax
import jax.numpy as jnp
from jax import lax, jit, vmap, random
import equinox as eqx

from flow.utils import divergence_jvp, divergence_hutchinson
from flow.integrators import get_integrator

class Flow(eqx.Module):
    """
    Implements a deterministic Continuous Normalizing Flow (CNF).

    A CNF models a complex target probability distribution p_1 by transforming samples
    from a simple base distribution p_0 (e.g., a Gaussian) through a path defined by an
    ordinary differential equation (ODE).

    The core of the flow is the velocity field `v(x, t)`, which defines the dynamics:
        dx/dt = v(x(t), t)

    - The 'forward' process integrates this ODE from t=0 to t=1, mapping a sample
      x_0 ~ p_0 to a sample x_1 ~ p_1.
    - The 'backward' process integrates a time-reversed ODE from t=1 to t=0,
      mapping a sample x_1 back to its corresponding x_0.

    A key feature of CNFs is the ability to compute the exact log-probability of a
    sample using the instantaneous change of variables formula:
        log p(x(t)) = log p(x(0)) - ∫_0^t div(v(x(s), s)) ds
    This class provides methods to perform both sampling (forward/backward) and
    exact likelihood calculation (push_forward_log_prob).
    """
    
    # 'velocity' is an Equinox Module (e.g. an MLP). JAX will automatically 
    # find and trace the arrays/parameters inside it.
    velocity: eqx.Module
    
    # We mark configuration integers/strings as static so JAX doesn't try to trace them.
    n_steps: int = eqx.field(static=True)
    integrator_name: str = eqx.field(static=True)
    integrator_kwargs: dict = eqx.field(static=True)

    def __init__(self, velocity, n_steps, integrator_name='rk4', **integrator_kwargs):
        self.velocity = velocity
        self.n_steps = n_steps
        self.integrator_name = integrator_name
        self.integrator_kwargs = integrator_kwargs

    def _get_solver(self, direction='forward'):
        """
        Helper to instantiate the integrator with the *current* velocity parameters.
        """
        integrator_fn = get_integrator(self.integrator_name)
        
        if direction == 'forward':
            # Solve dx/dt = v(x, t)
            # We pass self.velocity (the callable module) directly.
            return integrator_fn(self.velocity, **self.integrator_kwargs)
        else:
            # Solve dx/dt = -v(x, 1-t)
            # Use tree_map to handle PyTree states (e.g. dicts)
            return integrator_fn(lambda x, t: jax.tree_util.tree_map(lambda v: -v, self.velocity(x, 1.0 - t)), **self.integrator_kwargs)

    # --- Core Methods ---

    def forward(self, x_0):
        solver = self._get_solver('forward')
        return solver.solve(x_0, 0.0, 1.0, self.n_steps, None)

    def backward(self, y_T):
        solver = self._get_solver('backward')
        return solver.solve(y_T, 0.0, 1.0, self.n_steps, None)

    def forward_trajectory(self, x_0):
        solver = self._get_solver('forward')
        return solver.trajectory(x_0, 0.0, 1.0, self.n_steps, None)

    def backward_trajectory(self, y_T):
        solver = self._get_solver('backward')
        return solver.trajectory(y_T, 0.0, 1.0, self.n_steps, None)

    def accumulate(self, xts, ts, div_fn=None):
        """
        Integrates the divergence along the trajectory with the trapezoid rule on the
        trajectory grid (second order in the step size; n_steps + 1 divergence evaluations).
        Returns -int div v dt, the change in log-density along the path.
        """
        dt = ts[1] - ts[0]
        
        # Default to exact divergence (JVP) if no specific estimator provided
        if div_fn is None:
            # divergence_jvp usually expects (func, argnums). 
            # self.velocity is the func.
            exact_div = divergence_jvp(self.velocity, argnums=0)
            # Wrap to match signature expected by loop: (x, t, step_idx)
            div_fn = lambda x, t, step_idx=None: exact_div(x, t)

        def _body(n, acc_p_t):
            x_n = jax.tree_util.tree_map(lambda a: a[n], xts)   # array or PyTree state
            div = div_fn(x_n, ts[n], step_idx=n)
            weight = jnp.where((n == 0) | (n == self.n_steps), 0.5, 1.0)
            return acc_p_t + weight * dt * div
        
        acc_p_t = lax.fori_loop(0, self.n_steps + 1, _body, 0.0)
        return -acc_p_t

    def push_forward_log_prob(self, logp_x_0, x_0, key=None):
        r"""
        log p(x_1) = - \int_0^1 \nabla \cdot v(x, t) dt + log p(x_0)
        """
        xts = self.forward_trajectory(x_0)
        ts = jnp.linspace(0, 1, self.n_steps + 1)
        
        div_fn = None
        if key is not None:
            # Create Hutchinson estimator using the current velocity
            hutch_div = divergence_hutchinson(self.velocity, argnums=0)
            # Bind the key. step_idx is passed by accumulate's loop.
            div_fn = lambda x, t, step_idx: hutch_div(x, t, key=key, step_idx=step_idx)

        acc = self.accumulate(xts, ts, div_fn=div_fn)
        return acc + logp_x_0, jax.tree_util.tree_map(lambda a: a[-1], xts)
    
    def push_backward_log_prob(self, logp_x_1, x_1, key=None):
        xts = self.backward_trajectory(x_1)
        ts = jnp.linspace(1, 0, self.n_steps + 1)
        
        div_fn = None
        if key is not None:
            hutch_div = divergence_hutchinson(self.velocity, argnums=0)
            div_fn = lambda x, t, step_idx: hutch_div(x, t, key=key, step_idx=step_idx)

        acc = self.accumulate(xts, ts, div_fn=div_fn)
        return acc + logp_x_1, jax.tree_util.tree_map(lambda a: a[-1], xts)
    
    def push_forward(self, p_x_0, x_0):
        # Convenience method (work in log space then exp)
        log_p_x_1, x_1 = self.push_forward_log_prob(jnp.log(p_x_0), x_0)
        return jnp.exp(log_p_x_1), x_1

    def push_backward(self, p_x_1, x_1):
        log_p_x_0, x_0 = self.push_backward_log_prob(jnp.log(p_x_1), x_1)
        return jnp.exp(log_p_x_0), x_0

    # --- Batched / Jitted Entry Points ---
    
    # Note on vmap in Equinox:
    # When we vmap(self.method), JAX captures 'self'.
    # Because 'self' is a PyTree (an Equinox Module), JAX treats the weights
    # inside it as captured variables. Since we are NOT vmapping over the weights,
    # JAX broadcasts them across the batch. This is exactly what we want.

    @jit
    def b_forward(self, x_0):
        return vmap(self.forward)(x_0)

    @jit
    def b_backward(self, x_1):
        return vmap(self.backward)(x_1)

    @jit
    def b_forward_trajectory(self, x_0):
        return vmap(self.forward_trajectory)(x_0)

    @jit
    def b_backward_trajectory(self, x_1):
        return vmap(self.backward_trajectory)(x_1)

    @jit
    def b_push_forward_log_prob(self, logp_x_0, x_0, key=None):
        if key is not None:
            # Split keys for the batch so every sample gets different noise
            keys = random.split(key, x_0.shape[0])
            return vmap(self.push_forward_log_prob)(logp_x_0, x_0, keys)
        else:
            # Map over logp and x, broadcast None for key
            return vmap(self.push_forward_log_prob)(logp_x_0, x_0, None)

    @jit
    def b_push_backward_log_prob(self, logp_x_1, x_1, key=None):
        if key is not None:
            keys = random.split(key, x_1.shape[0])
            return vmap(self.push_backward_log_prob)(logp_x_1, x_1, keys)
        else:
            return vmap(self.push_backward_log_prob)(logp_x_1, x_1, None)

    @jit
    def b_push_forward(self, p_x_0, x_0):
        return vmap(self.push_forward)(p_x_0, x_0)

    @jit
    def b_push_backward(self, p_x_1, x_1):
        return vmap(self.push_backward)(p_x_1, x_1)