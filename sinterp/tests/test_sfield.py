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

@pytest.mark.parametrize("interpolator_class", [LinearDeterministic, TrigonometricVPStochastic])
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

@pytest.mark.parametrize("loss_name", ["loss_b", "loss_v", "loss_eta0", "loss_eta1", "loss_etaz"])
def test_pytree_loss_is_sum_of_leaf_losses(sample_data, loss_name):
    """For dict states each loss is the sum over leaves of the per-leaf loss (tree_reduce),
    so a structured field weighs every block. Catches reducing only the first leaf or
    averaging the leaves together. (Correctness of the loss *targets* is checked in
    test_sfield_oracle.py.)"""
    field = TwoSidedField_v_etaz(TrigonometricVPStochastic(), sample_data["v_model"], sample_data["eta_z_model"])
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(1), 3)
    shapes = {"a": (4,), "b": (2,)}
    x = {n: jax.random.normal(jax.random.fold_in(k1, i), s) for i, (n, s) in enumerate(shapes.items())}
    y = {n: jax.random.normal(jax.random.fold_in(k2, i), s) + 2.0 for i, (n, s) in enumerate(shapes.items())}
    e = {n: jax.random.normal(jax.random.fold_in(k3, i), s) for i, (n, s) in enumerate(shapes.items())}
    t = 0.3
    total = getattr(field, loss_name)(x, y, e, t)
    per_leaf = sum(getattr(field, loss_name)(x[n], y[n], e[n], t) for n in shapes)
    assert total.shape == ()
    assert jnp.allclose(total, per_leaf, rtol=1e-5)


def test_total_loss_is_sum_of_the_two_component_losses(sample_data):
    """Each field's total_loss is the sum of the losses of the two quantities it learns."""
    from sinterp.stochasticfield import TwoSidedField_b_s, TwoSidedField_x0_x1, TwoSidedField_b_eta1
    interp = TrigonometricVPStochastic()
    m1, m2 = sample_data["v_model"], sample_data["eta_z_model"]
    x, y, e = (jax.random.normal(jax.random.PRNGKey(i), (4,)) for i in range(3))
    t = 0.3
    cases = [
        (TwoSidedField_v_etaz(interp, m1, m2), ("loss_v", "loss_etaz")),
        (TwoSidedField_b_s(interp, m1, m2), ("loss_b", "loss_s")),
        (TwoSidedField_x0_x1(interp, m1, m2), ("loss_eta0", "loss_eta1")),
        (TwoSidedField_b_eta1(interp, m1, m2), ("loss_b", "loss_eta1")),
        (OneSidedField_b(OneSidedLinear(), m1), ("loss_b",)),
        (OneSidedField_eta1(OneSidedLinear(), m1), ("loss_eta1",)),
    ]
    for field, names in cases:
        want = sum(getattr(field, n)(x, y, e, t) for n in names)
        assert jnp.allclose(field.total_loss(x, y, e, t), want), type(field).__name__


# --- PyTree (block) states: every field class, every derived quantity ---

def _two_model_fields(interp):
    from sinterp.stochasticfield import TwoSidedField_b_s, TwoSidedField_x0_x1, TwoSidedField_b_eta1
    a, b = MockVModel(scale=1.5), MockEtaZModel(scale=0.5)
    return {
        "v_etaz": TwoSidedField_v_etaz(interp, a, b),
        "b_s": TwoSidedField_b_s(interp, a, b),
        "x0_x1": TwoSidedField_x0_x1(interp, a, b),
        "b_eta1": TwoSidedField_b_eta1(interp, a, b),
    }


@pytest.mark.parametrize("name", ["v_etaz", "b_s", "x0_x1", "b_eta1"])
def test_dict_state_equals_per_leaf_arrays(name):
    """A dict (block) state must give, leaf by leaf, exactly what the same field gives on each
    leaf as an array: the models act leafwise, so every derived quantity (b, v, s, eta0, eta1,
    etaz) is leafwise too. Catches plain arithmetic on PyTrees (TypeError) and any mixing of
    leaves."""
    interp = TrigonometricVPStochastic()
    field = _two_model_fields(interp)[name]
    k1, k2 = jax.random.split(jax.random.PRNGKey(3))
    x = {"a": jax.random.normal(k1, (4,)), "b": jax.random.normal(k2, (2,))}
    t = 0.37
    for getter in ("get_velocity_b", "get_velocity_v", "get_score_s",
                   "get_denoiser_eta0", "get_denoiser_eta1", "get_denoiser_etaz"):
        out = getattr(field, getter)(x, t)
        assert set(out) == {"a", "b"}, getter
        for leaf in ("a", "b"):
            want = getattr(field, getter)(x[leaf], t)
            assert jnp.allclose(out[leaf], want, atol=1e-6), (getter, leaf)
