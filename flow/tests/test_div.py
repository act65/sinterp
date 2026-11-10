import pytest
import jax
import jax.numpy as jnp

from flow.utils import divergence_jvp, divergence_hutchinson

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

def test_hutchinson_vs_exact_large_n():
    """
    Tests that Hutchinson divergence estimator is close to exact divergence 
    for a high-dimensional problem.
    """
    key = jax.random.PRNGKey(42)
    d = 1000
    x = jax.random.normal(key, (d,))
    
    # Define a function with non-trivial divergence
    # f(x) = x * sin(x)
    # J_ij = delta_ij * (sin(x_i) + x_i cos(x_i))
    # Div = sum(sin(x) + x cos(x))
    def fn(x):
        return x * jnp.sin(x)
        
    exact_div = divergence_jvp(fn)(x)
    
    # Hutchinson estimator
    # We average over a few samples to reduce variance for the test
    n_samples = 500
    keys = jax.random.split(key, n_samples)
    
    hutch_fn = divergence_hutchinson(fn)
    
    # vmap over keys
    hutch_divs = jax.vmap(lambda k: hutch_fn(x, key=k))(keys)
    mean_hutch_div = jnp.mean(hutch_divs)
    
    # For d=1000, relative error of single sample is ~sqrt(2/1000) ~ 4.5%
    # With 100 samples, standard error of mean is 4.5% / 10 = 0.45%
    # So they should be very close.
    
    # Check relative error
    rel_error = jnp.abs(mean_hutch_div - exact_div) / jnp.abs(exact_div)
    assert rel_error < 0.05, f"Hutchinson divergence relative error {rel_error} too high"