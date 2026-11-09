import jax
import jax.numpy as jnp
import equinox as eqx
from sinterp.field_factory import field_factory, make_default_net_constructor
from sinterp.interpolants import OneSidedLinear

def test_field_factory_dict():
    interpolator = OneSidedLinear()
    
    # Mock init_net that returns a simple lambda as a "network"
    # This one handles dict/PyTree by concatenating/splitting leaves
    def mock_init_net(n_outputs):
        class MockNet(eqx.Module):
            dim: int
            def __init__(self, dim):
                self.dim = dim
            def __call__(self, x, t, *args):
                # Returns a dict where each leaf has n_outputs * original_leaf_dim
                return jax.tree_util.tree_map(
                    lambda leaf: jnp.concatenate([leaf] * self.dim, axis=-1),
                    x
                )
        return MockNet(n_outputs)

    net_constructor = make_default_net_constructor(mock_init_net)
    
    # Dict input
    x = {"a": jnp.ones((10, 2)), "b": jnp.zeros((10, 3))}
    t = jnp.array([0.5])

    # Test OneSidedField_eta1 with dict
    field = field_factory("OneSidedField_eta1", net_constructor, interpolator)
    print(f"Created {type(field).__name__} with dict support")
    out = field.get_denoiser_eta1(x, t)
    assert out["a"].shape == x["a"].shape
    assert out["b"].shape == x["b"].shape
    print("OneSidedField_eta1 dict check passed")

    # Test TwoSidedField_b_eta1 with dict
    field2 = field_factory("TwoSidedField_b_eta1", net_constructor, interpolator)
    print(f"Created {type(field2).__name__} with dict support")
    b = field2.get_velocity_b(x, t)
    eta1 = field2.get_denoiser_eta1(x, t)
    assert b["a"].shape == x["a"].shape
    assert b["b"].shape == x["b"].shape
    assert eta1["a"].shape == x["a"].shape
    assert eta1["b"].shape == x["b"].shape
    print("TwoSidedField_b_eta1 dict check passed")

if __name__ == "__main__":
    test_field_factory_dict()
