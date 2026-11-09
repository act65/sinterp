
import jax.numpy as jnp
from jax import random
import pytest
from sinterp import exact_si_gmm, exact_si_gaussian
from sinterp.interpolants import LinearDeterministic, LinearStochastic

@pytest.fixture
def key():
    return random.PRNGKey(0)

def test_g_v_gmm_consistency(key):
    """
    Check that the exact gaussian SI gives results equal to exact GMM with 1 mode.
    """
    m1 = jnp.array([0.0])
    m2 = jnp.array([0.5])
    c1 = jnp.array([[1.0]])
    c2 = jnp.array([[0.5]])
    w1 = jnp.array([1.0])
    w2 = jnp.array([1.0])

    interp = LinearDeterministic()

    p0, v0, s0 = exact_si_gaussian.construct_p_b_s(m1, m2, c1, c2, interp)
    p1, v1, s1 = exact_si_gmm.construct_p_b_s(interp, [m1], [m2], [c1], [c2], w1, w2)

    for _ in range(10):
        key, subkey = random.split(key)
        z = random.uniform(subkey, shape=m1.shape) * 10 - 5
        key, subkey = random.split(key)
        t = random.uniform(subkey, ())

        assert jnp.isclose(p0(z, t), p1(z, t), atol=1e-5)
        assert jnp.allclose(v0(z, t), v1(z, t), atol=1e-5)
        assert jnp.allclose(s0(z, t), s1(z, t), atol=1e-5)

def test_boundary_conditions(key):
    """
    Check that at t=0 we recover source distribution and at t=1 we recover target.
    """
    m1s = [jnp.array([0.0])]
    c1s = [jnp.array([[1.0]])]
    w1s = jnp.array([1.0])
    
    m2s = [jnp.array([-2.0]), jnp.array([2.0])]
    c2s = [jnp.array([[0.25]]), jnp.array([[0.25]])]
    w2s = jnp.array([0.5, 0.5])
    
    interp = LinearDeterministic()
    p, v, s = exact_si_gmm.construct_p_b_s(interp, m1s, m2s, c1s, c2s, w1s, w2s)
    
    # Check t=0
    z = jnp.array([0.0])
    # Source is N(0, 1). pdf(0) = 1/sqrt(2pi) approx 0.3989
    expected_p0 = 0.39894228
    assert jnp.isclose(p(z, 0.0), expected_p0, atol=1e-4)
    
    # Check t=1
    # Target is 0.5*N(-2, 0.25) + 0.5*N(2, 0.25)
    # At z=2, contribution from N(2, 0.25) dominates.
    # N(2; 2, 0.25) = 1/sqrt(2pi*0.25) = 1/sqrt(pi/2) = sqrt(2/pi) approx 0.7978
    # Weighted by 0.5 -> 0.3989
    # Contribution from N(-2, 0.25) at z=2 is negligible (4 sigmas away)
    expected_p1 = 0.39894228
    assert jnp.isclose(p(jnp.array([2.0]), 1.0), expected_p1, atol=1e-4)

def test_stochastic_interpolant(key):
    """
    Check that stochastic interpolant runs without error and produces valid output.
    """
    m1s = [jnp.array([0.0])]
    c1s = [jnp.array([[1.0]])]
    w1s = jnp.array([1.0])
    
    m2s = [jnp.array([2.0])]
    c2s = [jnp.array([[1.0]])]
    w2s = jnp.array([1.0])
    
    # Use LinearStochastic which has non-zero gamma
    interp = LinearStochastic()
    # Pass explicit noise covariance (identity)
    ce = jnp.eye(1)
    p, v, s = exact_si_gmm.construct_p_b_s(interp, m1s, m2s, c1s, c2s, w1s, w2s, ce=ce)
    
    z = jnp.array([1.0])
    t = 0.5
    
    val_p = p(z, t)
    val_v = v(z, t)
    val_s = s(z, t)
    
    assert not jnp.isnan(val_p)
    assert not jnp.isnan(val_v).any()
    assert not jnp.isnan(val_s).any()
    
    # Check that gamma is actually being used (indirectly, by checking if results differ from deterministic if we forced gamma=0)
    # But here we just want to ensure it runs.

def test_stochastic_vs_deterministic(key):
    """
    Verify that a stochastic interpolant produces different results (e.g. higher variance/lower peak density)
    than a deterministic one, confirming that gamma is being used.
    """
    m1s = [jnp.array([0.0])]
    c1s = [jnp.array([[1.0]])]
    w1s = jnp.array([1.0])
    
    m2s = [jnp.array([0.0])] # Same mean to isolate variance effects
    c2s = [jnp.array([[1.0]])]
    w2s = jnp.array([1.0])
    
    # Deterministic: LinearDeterministic (gamma=0)
    # Path should be constant N(0, 1) if alpha(t)=1-t, beta(t)=t, gamma=0
    # Wait, LinearDeterministic: alpha=1-t, beta=t. 
    # c_t = (1-t)^2 + t^2. At t=0.5, c_t = 0.25 + 0.25 = 0.5. 
    # So it contracts and expands. It's not constant variance.
    interp_det = LinearDeterministic()
    p_det, _, _ = exact_si_gmm.construct_p_b_s(interp_det, m1s, m2s, c1s, c2s, w1s, w2s)
    
    # Stochastic: LinearStochastic
    # alpha, beta, gamma are different.
    # But let's check if the density value at the mean (z=0) is different.
    interp_stoch = LinearStochastic()
    # Need to pass ce for stochastic
    ce = jnp.eye(1)
    p_stoch, _, _ = exact_si_gmm.construct_p_b_s(interp_stoch, m1s, m2s, c1s, c2s, w1s, w2s, ce=ce)
    
    z = jnp.array([0.0])
    t = 0.5
    
    val_det = p_det(z, t)
    val_stoch = p_stoch(z, t)
    
    # They should be different
    assert not jnp.isclose(val_det, val_stoch, atol=1e-5)
    
    # Specifically, LinearStochastic at t=0.5:
    # alpha=0, beta=0, gamma=1 (from code: gamma = 2t for t<0.5? No, let's check interpolants.py)
    # LinearStochastic:
    # t < 0.5: alpha=1-2t, beta=0, gamma=2t
    # t=0.5: alpha=0, beta=0, gamma=1.
    # So at t=0.5, x_t = gamma(t)*z ~ N(0, 1). Variance is 1.
    #
    # LinearDeterministic at t=0.5:
    # alpha=0.5, beta=0.5, gamma=0.
    # Variance = 0.5^2*1 + 0.5^2*1 = 0.25 + 0.25 = 0.5.
    #
    # So Stochastic variance (1.0) > Deterministic variance (0.5).
    # Therefore, Stochastic peak density (at 0) should be LOWER than Deterministic peak density.
    # pdf ~ 1/sqrt(var). 
    
    assert val_stoch < val_det