"""field_factory / make_default_net_constructor claims.

A two-output net's output is split in half along the last axis: the FIRST half is the first
quantity in the field's name (b in b_eta1 / b_s, v in v_etaz, eta0 in x0_x1) and the SECOND
half the second. The mock net emits two *different* halves, so a swapped split or a wrong
mapping fails (the previous mocks concatenated identical copies of x, which hid both)."""
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from sinterp.field_factory import field_factory, make_default_net_constructor
from sinterp.interpolants import OneSidedLinear, TrigonometricVPStochastic


def first(x):
    return 2.0 * x + 1.0


def second(x):
    return -3.0 * x + 0.5


class TwoHeadNet(eqx.Module):
    n_outputs: int = eqx.field(static=True)

    def __call__(self, x, t, *args):
        f = lambda leaf: (jnp.concatenate([first(leaf), second(leaf)], axis=-1)
                          if self.n_outputs == 2 else first(leaf))
        return jax.tree_util.tree_map(f, x)


net_constructor = make_default_net_constructor(lambda n_outputs: TwoHeadNet(n_outputs))

# field name -> (getter of the first half, getter of the second half)
TWO_OUTPUT = {
    "TwoSidedField_b_eta1": ("get_velocity_b", "get_denoiser_eta1"),
    "TwoSidedField_v_etaz": ("get_velocity_v", "get_denoiser_etaz"),
    "TwoSidedField_b_s": ("get_velocity_b", "get_score_s"),
    "TwoSidedField_x0_x1": ("get_denoiser_eta0", "get_denoiser_eta1"),
}
ONE_OUTPUT = {"OneSidedField_eta1": "get_denoiser_eta1", "OneSidedField_b": "get_velocity_b"}

STATES = {
    "array": jnp.linspace(-1.0, 1.0, 10).reshape(5, 2),
    "dict": {"a": jnp.linspace(-1.0, 1.0, 10).reshape(5, 2), "b": jnp.linspace(0.0, 2.0, 15).reshape(5, 3)},
}


@pytest.mark.parametrize("state", list(STATES))
@pytest.mark.parametrize("name", list(TWO_OUTPUT))
def test_two_output_split_maps_halves_to_named_quantities(name, state):
    x, t = STATES[state], 0.3
    field = field_factory(name, net_constructor, TrigonometricVPStochastic())
    g1, g2 = TWO_OUTPUT[name]
    out1, out2 = getattr(field, g1)(x, t), getattr(field, g2)(x, t)
    for o, f in ((out1, first), (out2, second)):
        jax.tree_util.tree_map(lambda got, leaf: (got.shape == leaf.shape) or pytest.fail("shape"), o, x)
        assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda got, leaf: jnp.allclose(got, f(leaf)), o, x))


@pytest.mark.parametrize("state", list(STATES))
@pytest.mark.parametrize("name", list(ONE_OUTPUT))
def test_one_output_fields_use_the_net_directly(name, state):
    x, t = STATES[state], 0.3
    field = field_factory(name, net_constructor, OneSidedLinear())
    out = getattr(field, ONE_OUTPUT[name])(x, t)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda got, leaf: jnp.allclose(got, first(leaf)), out, x))


def test_unknown_names_are_rejected():
    with pytest.raises(ValueError):
        field_factory("NoSuchField", net_constructor, OneSidedLinear())
    with pytest.raises(ValueError):
        net_constructor(n_outputs=3)
