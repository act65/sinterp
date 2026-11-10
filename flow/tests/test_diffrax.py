import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import diffrax
import pytest

from flow.flow import Flow

class FuncVelocity(eqx.Module):
    func: callable = eqx.field(static=True)
    def __call__(self, x, t):
        return self.func(x, t)

def test_diffrax_integration():
    d = 2
    key = random.PRNGKey(0)
    x = random.normal(key, (d,))
    
    # Simple velocity field
    v = FuncVelocity(lambda x, t: -x)
    
    # Test with default diffrax settings (Dopri5, ConstantStepSize by default in my wrapper)
    # Wait, my wrapper defaults to ConstantStepSize if not provided.
    # If I want adaptive, I should pass PIDController.
    
    flow = Flow(v, n_steps=100, integrator_name='diffrax')
    y = flow.forward(x)
    
    # Exact solution for dx/dt = -x is x(t) = x(0) * exp(-t)
    # At t=1, x(1) = x(0) * exp(-1)
    expected = x * jnp.exp(-1.0)
    
    assert jnp.allclose(y, expected, atol=1e-3)
    print("Diffrax basic integration test passed!")

def test_diffrax_adaptive():
    d = 2
    key = random.PRNGKey(0)
    x = random.normal(key, (d,))
    
    v = FuncVelocity(lambda x, t: -x)
    
    # Use adaptive step size
    controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
    flow = Flow(v, n_steps=0, integrator_name='diffrax', 
                stepsize_controller=controller, 
                solver=diffrax.Tsit5())
    
    y = flow.forward(x)
    expected = x * jnp.exp(-1.0)
    
    assert jnp.allclose(y, expected, atol=1e-6)
    print("Diffrax adaptive integration test passed!")

def test_diffrax_trajectory():
    d = 2
    key = random.PRNGKey(0)
    x = random.normal(key, (d,))
    
    v = FuncVelocity(lambda x, t: -x)
    n_steps = 10
    flow = Flow(v, n_steps=n_steps, integrator_name='diffrax')
    
    xts = flow.forward_trajectory(x)
    
    assert xts.shape == (n_steps + 1, d)
    
    ts = jnp.linspace(0, 1, n_steps + 1)
    expected_xts = jax.vmap(lambda t: x * jnp.exp(-t))(ts)
    
    assert jnp.allclose(xts, expected_xts, atol=1e-3)
    print("Diffrax trajectory test passed!")

def test_diffrax_log_prob():
    d = 2
    key = random.PRNGKey(0)
    x = random.normal(key, (d,))
    
    # Linear velocity: v(x, t) = a * x
    # div(v) = a * d
    # log p(x(1)) = log p(x(0)) - \int_0^1 (a * d) dt = log p(x(0)) - a * d
    a = 0.5
    v = FuncVelocity(lambda x, t: a * x)
    
    n_steps = 100
    controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
    flow = Flow(v, n_steps=n_steps, integrator_name='diffrax',
                stepsize_controller=controller)
    
    log_p0 = jnp.array(0.0)
    log_p1, y1 = flow.push_forward_log_prob(log_p0, x)
    
    expected_log_p1 = log_p0 - (a * d)
    
    assert jnp.allclose(log_p1, expected_log_p1, atol=1e-3)
    print("Diffrax log_prob test passed!")

if __name__ == "__main__":
    test_diffrax_integration()
    test_diffrax_adaptive()
    test_diffrax_trajectory()
    test_diffrax_log_prob()
