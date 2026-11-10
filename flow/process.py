import jax.numpy as jnp
from jax import jit, vmap

from flow.integrators import get_integrator

class Process():
    """
    A stochastic process (SDE) based on the stochastic interpolant framework.
    It simulates the forward and backward SDEs whose solutions have the same
    time-dependent density p(t) as the corresponding deterministic flow.
    """
    def __init__(self, b, s, epsilon, n_steps, integrator_name='euler-maruyama'):
        """
        Given an SDE integrator, construct functions to simulate trajectories.

        Args:
            b (callable): The velocity field b: (X, T) -> X.
            s (callable): The score field s: (X, T) -> X.
            epsilon (callable): A time-dependent diffusion coefficient epsilon: T -> R+.
            n_steps (int): The number of steps for the integrator.
            integrator_name (str): The name of the SDE integrator to use.
        """
        self.n_steps = n_steps
        self.b = b
        self.s = s
        self.epsilon = epsilon

        # Define forward and backward drifts and diffusions from the paper (Eq. 2.21, 2.23)
        self.drift_f = lambda x, t: self.b(x, t) + self.epsilon(t) * self.s(x, t)
        self.drift_b = lambda x, t: -(self.b(x, 1-t) - self.epsilon(1-t) * self.s(x, 1-t))
        
        self.diffusion_f = lambda t: jnp.sqrt(2 * self.epsilon(t))
        self.diffusion_b = lambda t: jnp.sqrt(2 * self.epsilon(1-t))

        integrator_class = get_integrator(integrator_name)
        
        self.integrator_f = integrator_class(self.drift_f, self.diffusion_f)
        self.integrator_b = integrator_class(self.drift_b, self.diffusion_b)

        # JIT the simulation methods
        self.forward = jit(self.forward)
        self.backward = jit(self.backward)
        self.b_forward = jit(vmap(self.forward, in_axes=(0, 0)))
        self.b_backward = jit(vmap(self.backward, in_axes=(0, 0)))

        # JIT the trajectory methods
        self.forward_trajectory = jit(self.forward_trajectory)
        self.backward_trajectory = jit(self.backward_trajectory)
        self.b_forward_trajectory = jit(vmap(self.forward_trajectory, in_axes=(0, 0)))
        self.b_backward_trajectory = jit(vmap(self.backward_trajectory, in_axes=(0, 0)))

    def forward(self, x_0, key):
        """Simulates the forward SDE from t=0 to t=1."""
        return self.integrator_f.solve(x_0, t0=0.0, t1=1.0, n_steps=self.n_steps, key=key)

    def backward(self, y_T, key):
        """Simulates the backward SDE from t=1 to t=0."""
        return self.integrator_b.solve(y_T, t0=0.0, t1=1.0, n_steps=self.n_steps, key=key)

    def forward_trajectory(self, x_0, key):
        """Returns the full trajectory of the forward SDE."""
        return self.integrator_f.trajectory(x_0, t0=0.0, t1=1.0, n_steps=self.n_steps, key=key)

    def backward_trajectory(self, y_T, key):
        """Returns the full trajectory of the backward SDE."""
        return self.integrator_b.trajectory(y_T, t0=0.0, t1=1.0, n_steps=self.n_steps, key=key)