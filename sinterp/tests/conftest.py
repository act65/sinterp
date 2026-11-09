import pytest
import jax
import jax.numpy as jnp

@pytest.fixture
def key():
    return jax.random.PRNGKey(42)

@pytest.fixture
def dummy_data(key):
    """Generates simple Gaussian blobs for testing."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    batch_size = 100
    dim = 2
    
    x0 = jax.random.normal(k1, (batch_size, dim))
    x1 = jax.random.normal(k2, (batch_size, dim)) + 5.0  # Shifted
    
    # Binary conditions (0.0 or 1.0)
    c0 = jax.random.bernoulli(k3, 0.5, (batch_size, 1)).astype(jnp.float32)
    c1 = jax.random.bernoulli(k4, 0.5, (batch_size, 1)).astype(jnp.float32)
    
    return x0, x1, c0, c1