import jax
import jax.numpy as jnp
import equinox as eqx
from sinterp.field_factory import field_factory, make_default_net_constructor
from sinterp.interpolants import OneSidedLinear

def test_field_factory():
    interpolator = OneSidedLinear()
    
    # Mock init_net that returns a simple lambda as a "network"
    # In reality this would be an eqx.Module
    def mock_init_net(n_outputs):
        class MockNet(eqx.Module):
            dim: int
            def __init__(self, dim):
                self.dim = dim
            def __call__(self, x, t, *args):
                return jnp.concatenate([x] * self.dim, axis=-1)
        return MockNet(n_outputs)

    net_constructor = make_default_net_constructor(mock_init_net)
    
    x = jnp.ones((10, 2))
    t = jnp.array([0.5])

    # Test OneSidedField_eta1
    field = field_factory("OneSidedField_eta1", net_constructor, interpolator)
    print(f"Created {type(field).__name__}")
    out = field.get_denoiser_eta1(x, t)
    assert out.shape == x.shape
    print("OneSidedField_eta1 check passed")

    # Test TwoSidedField_b_eta1
    field2 = field_factory("TwoSidedField_b_eta1", net_constructor, interpolator)
    print(f"Created {type(field2).__name__}")
    b = field2.get_velocity_b(x, t)
    eta1 = field2.get_denoiser_eta1(x, t)
    assert b.shape == x.shape
    assert eta1.shape == x.shape
    print("TwoSidedField_b_eta1 check passed")

if __name__ == "__main__":
    test_field_factory()
