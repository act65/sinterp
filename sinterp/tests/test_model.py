# test.py (Corrected)

import pytest
import jax
import jax.numpy as jnp

# Assume these are imported from your library, sinterp.model
from sinterp.model import (
    GenerativeModel,
    OneSidedFlowModel)
from sinterp.interpolants import (
    Interpolator,
    TrigonometricStochastic
)

# ===================================================================
# Correct Interpolator Definition for Testing
# ===================================================================

class OneSidedLinear(Interpolator):
    """
    Defines a one-sided interpolant: x_t = α(t)z + β(t)x₁
    This is a two-component system where the determinant of the
    coefficient matrix is non-zero, allowing for algebraic manipulation.
    """
    def __init__(self):
        super().__init__()

    def alpha(self, t):
        return 1.0 - t

    def beta(self, t):
        return t

    def gamma(self, t):
        # No third component in this specific one-sided formulation
        return 0.0 * t

# ===================================================================
# Fixtures: Reusable components for our tests
# ===================================================================

@pytest.fixture
def dummy_params():
    """Provides a sample parameter dictionary for the dummy models."""
    return {'decay': 0.5, 'drift': 0.1}

@pytest.fixture
def dummy_data():
    """Provides sample input data (x_t) and time (t)."""
    key = jax.random.PRNGKey(42)
    x_t = jax.random.normal(key, (10,))
    t = 0.25
    return x_t, t

# --- Models for Two-Sided Tests ---

@pytest.fixture
def dummy_eta_0_model():
    """A placeholder for a learned E[x_0 | x_t] network."""
    def model(x, t, params):
        return x * (1 - t) + params['drift']
    return model

@pytest.fixture
def dummy_eta_1_model():
    """A placeholder for a learned E[x_1 | x_t] network."""
    def model(x, t, params):
        return x * t - params['drift']
    return model

@pytest.fixture
def two_sided_model(dummy_eta_0_model, dummy_eta_1_model):
    """A GenerativeModel configured for a two-sided interpolant."""
    interpolator = TrigonometricStochastic()
    return GenerativeModel(
        interpolator=interpolator,
        eta_0_model=dummy_eta_0_model,
        eta_1_model=dummy_eta_1_model
    )

# --- Models for One-Sided Tests ---

@pytest.fixture
def dummy_velocity_model():
    """A placeholder for a single, learned velocity network b(t, x_t)."""
    def model(x, t, params):
        return -x * params['decay'] + (1 - t) * params['drift']
    return model

@pytest.fixture
def one_sided_model(dummy_velocity_model):
    """A OneSidedFlowModel configured with a single velocity network."""
    # Use the corrected, non-singular interpolator
    interpolator = OneSidedLinear()
    return OneSidedFlowModel(
        interpolator=interpolator,
        velocity_model=dummy_velocity_model
    )

# ===================================================================
# Test Cases
# ===================================================================

def test_two_sided_model_initialization(two_sided_model):
    """Tests that the two-sided model initializes correctly."""
    assert two_sided_model is not None
    assert isinstance(two_sided_model.interpolator, TrigonometricStochastic)
    assert two_sided_model.eta_z_model is None

def test_two_sided_model_outputs(two_sided_model, dummy_data, dummy_params):
    """
    Tests that the methods of the two-sided model produce outputs
    of the correct shape.
    """
    x_t, t = dummy_data
    
    velocity = two_sided_model.get_velocity(x_t, t, params_0=dummy_params, params_1=dummy_params)
    score = two_sided_model.get_score(x_t, t, params_0=dummy_params, params_1=dummy_params)
    denoised = two_sided_model.denoise_x0(x_t, t, params_0=dummy_params, params_1=dummy_params)

    assert velocity.shape == x_t.shape
    assert score.shape == x_t.shape
    assert denoised.shape == x_t.shape

def test_two_sided_model_conversions(two_sided_model, dummy_data, dummy_params):
    """
    Tests the self-consistency of the velocity-to-score and score-to-velocity
    conversions for the two-sided model.
    """
    x_t, t = dummy_data
    
    original_velocity = two_sided_model.get_velocity(x_t, t, params_0=dummy_params, params_1=dummy_params)
    original_score = two_sided_model.get_score(x_t, t, params_0=dummy_params, params_1=dummy_params)

    converted_score = two_sided_model.velocity_to_score(original_velocity, x_t, t, params_0=dummy_params, params_1=dummy_params)
    reverted_velocity = two_sided_model.score_to_velocity(converted_score, x_t, t, params_0=dummy_params, params_1=dummy_params)

    assert jnp.allclose(converted_score, original_score, atol=1e-5)
    assert jnp.allclose(reverted_velocity, original_velocity, atol=1e-5)


def test_one_sided_model_initialization(one_sided_model):
    """Tests that the one-sided model initializes correctly."""
    assert one_sided_model is not None
    assert isinstance(one_sided_model.interpolator, OneSidedLinear)
    assert one_sided_model.velocity_model is not None

def test_one_sided_model_velocity(one_sided_model, dummy_data, dummy_params):
    """
    Tests that get_velocity on the specialized model correctly calls the
    underlying velocity network.
    """
    x_t, t = dummy_data
    
    # This test now calls the overridden get_velocity method
    model_velocity = one_sided_model.get_velocity(x_t, t, velocity_params=dummy_params)
    
    direct_velocity = one_sided_model.velocity_model(x_t, t, dummy_params)
    
    assert jnp.allclose(model_velocity, direct_velocity)

def test_one_sided_algebraic_consistency(one_sided_model, dummy_data, dummy_params):
    """
    The most critical test for the one-sided model.
    It verifies that the algebraically derived eta values satisfy the
    fundamental constraint equation: x_t ≈ α(t)η_z + β(t)η₁.
    This test now passes because the OneSidedLinear fixture is non-singular.
    """
    x_t, t = dummy_data
    
    eta_1 = one_sided_model.eta_1_from_velocity(x_t, t, velocity_params=dummy_params)
    eta_z = one_sided_model.eta_z_from_velocity(x_t, t, velocity_params=dummy_params)

    alpha_t = one_sided_model.interpolator.alpha(t)
    beta_t = one_sided_model.interpolator.beta(t)

    reconstructed_x_t = (alpha_t * eta_z) + (beta_t * eta_1)

    assert jnp.allclose(reconstructed_x_t, x_t, atol=1e-6)