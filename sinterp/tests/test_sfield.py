import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from sinterp.stochasticfield import (
    StochasticField, 
    TwoSidedField, 
    TwoSidedField_v_etaz, 
    OneSidedField_b, 
    OneSidedField_eta1
)

from sinterp.interpolants import (
    Interpolator,
    LinearDeterministic,
    LinearStochastic,
    TrigonometricVPStochastic,
    OneSidedLinear,
    OneSidedSBDM
)

# --- Mock Models ---
class MockVModel(eqx.Module):
    scale: float
    def __call__(self, x_t, t):
        return jax.tree_util.tree_map(lambda x: self.scale * jnp.sin(t * jnp.pi) * x, x_t)

class MockEtaZModel(eqx.Module):
    scale: float
    def __call__(self, x_t, t):
        return jax.tree_util.tree_map(lambda x: self.scale * (1.0 - t) * x, x_t)

class MockBModel(eqx.Module):
    scale: float
    def __call__(self, x_t, t):
        return jax.tree_util.tree_map(lambda x: self.scale * jnp.cos(t * jnp.pi) * x, x_t)

# --- Fixtures ---
@pytest.fixture
def key():
    return jax.random.PRNGKey(0)

@pytest.fixture
def sample_data(key):
    x_shape = (4,)
    return {
        "x_t": jax.random.normal(key, shape=x_shape),
        "x_t_dict": {"a": jax.random.normal(key, shape=x_shape), "b": jax.random.normal(key, shape=(2,))},
        "t": 0.5,
        "v_model": MockVModel(scale=1.5),
        "eta_z_model": MockEtaZModel(scale=0.5),
        "b_model": MockBModel(scale=2.0)
    }

# --- Tests for TwoSidedField ---

@pytest.mark.parametrize("interpolator_class", [LinearDeterministic, LinearStochastic, TrigonometricVPStochastic])
@pytest.mark.parametrize("x_t_key", ["x_t", "x_t_dict"])
def test_twosided_get_learned_components(sample_data, interpolator_class, x_t_key):
    interpolator = interpolator_class()
    field = TwoSidedField_v_etaz(interpolator, sample_data["v_model"], sample_data["eta_z_model"])
    
    x_t, t = sample_data[x_t_key], sample_data["t"]
    v_val = field.get_velocity_v(x_t, t)
    expected_v = sample_data["v_model"](x_t, t)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.allclose, v_val, expected_v))

@pytest.mark.parametrize("interpolator_class", [LinearDeterministic])
@pytest.mark.parametrize("x_t_key", ["x_t", "x_t_dict"])
def test_twosided_derived_quantities(sample_data, interpolator_class, x_t_key):
    interpolator = interpolator_class()
    field = TwoSidedField_v_etaz(interpolator, sample_data["v_model"], sample_data["eta_z_model"])
    
    x_t, t = sample_data[x_t_key], sample_data["t"]

    v_val = field.get_velocity_v(x_t, t)
    eta_z_val = field.get_denoiser_etaz(x_t, t)
    b_val = field.get_velocity_b(x_t, t)
    s_val = field.get_score_s(x_t, t)
    eta_0_val = field.get_denoiser_eta0(x_t, t)
    eta_1_val = field.get_denoiser_eta1(x_t, t)

    # 1. Test velocity b
    gamma_dot_t = interpolator.dgammadt(t)
    expected_b = jax.tree_util.tree_map(lambda _v, _ez: _v + gamma_dot_t * _ez, v_val, eta_z_val)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.allclose, b_val, expected_b))

    # 2. Test score s
    gamma_t = interpolator.gamma(t)
    if abs(gamma_t) > 1e-9:
        expected_s = jax.tree_util.tree_map(lambda _ez: -_ez / gamma_t, eta_z_val)
        assert jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.allclose, s_val, expected_s))

    # 3. Test consistency (TwoSided: x_t = alpha*eta0 + beta*eta1 + gamma*etaz)
    # Note: In TwoSidedField logic, x_t - gamma*etaz = alpha*eta0 + beta*eta1
    alpha_t, beta_t = interpolator.alpha(t), interpolator.beta(t)
    reconstructed_x_geom = jax.tree_util.tree_map(lambda _e0, _e1: alpha_t * _e0 + beta_t * _e1, eta_0_val, eta_1_val)
    
    diff = jax.tree_util.tree_map(lambda _xt, _ez: _xt - gamma_t * _ez, x_t, eta_z_val)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda d, r: jnp.allclose(d, r, atol=1e-5), diff, reconstructed_x_geom))


# --- Tests for OneSidedField ---

@pytest.mark.parametrize("interpolator_class", [OneSidedLinear, OneSidedSBDM])
@pytest.mark.parametrize("x_t_key", ["x_t", "x_t_dict"])
def test_onesided_derived_quantities(sample_data, interpolator_class, x_t_key):
    """
    Tests the calculation of derived quantities for the one-sided case.
    Considers OneSided logic: alpha=0, gamma handles the noise component.
    """
    interpolator = interpolator_class()
    field = OneSidedField_b(interpolator, sample_data["b_model"])

    x_t, t = sample_data[x_t_key], sample_data["t"]

    b_val = field.get_velocity_b(x_t, t)
    eta_0_val = field.get_denoiser_eta0(x_t, t)
    eta_1_val = field.get_denoiser_eta1(x_t, t)
    v_val = field.get_velocity_v(x_t, t)
    s_val = field.get_score_s(x_t, t)
    eta_z_val = field.get_denoiser_etaz(x_t, t)

    # 1. Test consistency of denoisers
    # OneSided Logic: x_t = gamma*eta_z + beta*eta_1 (eta0 is zero/irrelevant)
    alpha_t, beta_t = interpolator.alpha(t), interpolator.beta(t)
    gamma_t = interpolator.gamma(t)
    
    alpha_dot_t, beta_dot_t = interpolator.dalphadt(t), interpolator.dbetadt(t)
    gamma_dot_t = interpolator.dgammadt(t)

    # Check that eta0 is zero (as alpha is zero)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda _e0: jnp.allclose(_e0, 0.0), eta_0_val))
    assert jnp.isclose(alpha_t, 0.0)

    # Reconstruct x_t using the OneSided components
    reconstructed_x_t = jax.tree_util.tree_map(lambda _ez, _e1: gamma_t * _ez + beta_t * _e1, eta_z_val, eta_1_val)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda xt, rxt: jnp.allclose(xt, rxt, atol=1e-5), x_t, reconstructed_x_t))

    # Reconstruct b using the OneSided components
    reconstructed_b = jax.tree_util.tree_map(lambda _ez, _e1: gamma_dot_t * _ez + beta_dot_t * _e1, eta_z_val, eta_1_val)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda bv, rb: jnp.allclose(bv, rb, atol=1e-5), b_val, reconstructed_b))

    # 2. Test velocity v: v = beta' * eta_1
    expected_v = jax.tree_util.tree_map(lambda _e1: beta_dot_t * _e1, eta_1_val)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.allclose, v_val, expected_v))

    # 3. Test score s: s = -eta_z / gamma (OneSided uses gamma for noise)
    if abs(gamma_t) > 1e-9:
        expected_s = jax.tree_util.tree_map(lambda _ez: -_ez / gamma_t, eta_z_val)
        assert jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.allclose, s_val, expected_s))

@pytest.mark.parametrize("interpolator_class", [LinearDeterministic])
@pytest.mark.parametrize("x_t_key", ["x_t", "x_t_dict"])
def test_loss_methods(sample_data, interpolator_class, x_t_key):
    interpolator = interpolator_class()
    field = TwoSidedField_v_etaz(interpolator, sample_data["v_model"], sample_data["eta_z_model"])
    
    x_t, t = sample_data[x_t_key], sample_data["t"]
    x = jax.tree_util.tree_map(jnp.ones_like, x_t)
    y = jax.tree_util.tree_map(lambda val: val * 2, x)
    e = jax.tree_util.tree_map(lambda val: jax.random.normal(jax.random.PRNGKey(1), shape=val.shape), x_t)

    assert field.loss_b(x, y, e, t).shape == ()
    assert field.loss_v(x, y, e, t).shape == ()
    assert field.loss_s(x, y, e, t).shape == ()
    assert field.loss_eta0(x, y, e, t).shape == ()
    assert field.loss_eta1(x, y, e, t).shape == ()
    assert field.loss_etaz(x, y, e, t).shape == ()

@pytest.mark.parametrize("interpolator_class", [LinearDeterministic])
@pytest.mark.parametrize("x_t_key", ["x_t", "x_t_dict"])
def test_total_loss(sample_data, interpolator_class, x_t_key):
    interpolator = interpolator_class()
    
    field_v_etaz = TwoSidedField_v_etaz(interpolator, sample_data["v_model"], sample_data["eta_z_model"])
    field_b = OneSidedField_b(interpolator, sample_data["b_model"])
    
    x_t, t = sample_data[x_t_key], sample_data["t"]
    x = jax.tree_util.tree_map(jnp.ones_like, x_t)
    y = jax.tree_util.tree_map(lambda val: val * 2, x)
    e = jax.tree_util.tree_map(lambda val: jax.random.normal(jax.random.PRNGKey(1), shape=val.shape), x_t)
    
    assert field_v_etaz.total_loss(x, y, e, t).shape == ()
    assert field_b.total_loss(x, y, e, t).shape == ()
