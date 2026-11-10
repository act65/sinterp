import pytest
import jax
import jax.numpy as jnp

from flow.utils import divergence_jvp

# --- Pytest Tests ---

@pytest.mark.parametrize("div_func", [divergence_jvp])
def test_linear_divergence(div_func):
    """Tests a simple linear function f(x) = 2*x."""
    def linear_fn(x):
        return 2.0 * x

    x = jnp.array([1.0, 2.0, 3.0])
    expected_div = 6.0
    
    div = div_func(linear_fn)(x)
    assert jnp.allclose(div, expected_div)

@pytest.mark.parametrize("div_func", [divergence_jvp])
def test_nonlinear_divergence(div_func):
    """Tests a non-linear function f(x) = x**2."""
    def nonlinear_fn(x):
        return x**2

    x = jnp.array([1.0, 2.0, 3.0])
    expected_div = 12.0
    
    div = div_func(nonlinear_fn)(x)
    assert jnp.allclose(div, expected_div)

@pytest.mark.parametrize("div_func", [divergence_jvp])
def test_matrix_divergence(div_func):
    """Tests a function defined by a matrix multiplication f(x) = A @ x."""
    A = jnp.array([[1., 2., 3.],
                   [4., 5., 6.],
                   [7., 8., 9.]])
    def matrix_fn(x):
        return A @ x

    x = jnp.array([10.0, 11.0, 12.0])
    expected_div = 15.0
    
    div = div_func(matrix_fn)(x)
    assert jnp.allclose(div, expected_div)

@pytest.mark.parametrize("div_func", [divergence_jvp])
def test_multi_arg_divergence(div_func):
    """Tests a function with multiple arguments using argnums."""
    def multi_arg_fn(x, y):
        return x * y**2

    x = jnp.array([2.0, 3.0])
    y = jnp.array([4.0, 5.0])

    # Test divergence with respect to x (argnums=0)
    # Jacobian is diag(y**2) = diag([16, 25]). Trace = 16 + 25 = 41.
    div_wrt_x = div_func(multi_arg_fn, argnums=0)(x, y)
    assert jnp.allclose(div_wrt_x, 41.0)

    # Test divergence with respect to y (argnums=1)
    # Jacobian is diag(2*x*y) = diag([2*2*4, 2*3*5]) = diag([16, 30]). Trace = 16 + 30 = 46.
    div_wrt_y = div_func(multi_arg_fn, argnums=1)(x, y)
    assert jnp.allclose(div_wrt_y, 46.0)

@pytest.mark.parametrize("div_func", [divergence_jvp])
def test_reshaped_input_divergence(div_func):
    """Tests that the flattening wrapper works for non-vector inputs."""
    def image_fn(img):
        return jnp.sin(img)

    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    expected_div = jnp.sum(jnp.cos(x.ravel()))
    
    div = div_func(image_fn)(x)
    assert jnp.allclose(div, expected_div)

# The Hutchinson estimator (unbiasedness and Gaussian-probe variance) is tested with
# Monte-Carlo-standard-error tolerances in test_flow_claims.py.
