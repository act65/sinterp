import unittest

import jax.numpy as jnp

from sinterp.interpolants import interpolators

def check_boundaries(interp):
    t = jnp.linspace(0, 1, 100)

    a = interp.alpha(t)
    b = interp.beta(t)
    c = interp.gamma(t)

    assert jnp.isclose(a[0], 1.0, atol=1e-12)
    assert jnp.isclose(a[-1], 0.0, atol=1e-12)

    assert jnp.isclose(b[0], 0.0, atol=1e-12)
    assert jnp.isclose(b[-1], 1.0, atol=1e-12)

    assert jnp.isclose(c[0], 0.0, atol=1e-12)
    assert jnp.isclose(c[-1], 0.0, atol=1e-12)

class TestInterp(unittest.TestCase):
    def test_interp(self):
        for interp in interpolators:
            i = interp()
            check_boundaries(i)

    def test_grad(self):
        for interp in interpolators:
            t = jnp.linspace(0, 1, 100)
            i = interp()
            da = i.dalphadt(t[0])
            assert da.shape == ()

            shape = (28, 28, 1)

            di = i.didt(jnp.ones(shape), jnp.ones(shape), jnp.ones(shape), 0.0)
            assert di.shape == shape