"""Interpolant schedules: x_t = alpha(t) x0 + beta(t) x1 + gamma(t) z.

Claims:
* two-sided: I(0) = x0, I(1) = x1 (alpha: 1 -> 0, beta: 0 -> 1, gamma: 0 at both ends);
  one-sided: I(0) = z, I(1) = x1 (gamma: 1 -> 0, x0 ignored); ConstantNoise keeps gamma = sigma;
* the docstrings' variance-preserving claim alpha^2 + beta^2 + gamma^2 = 1 for the VP / trig
  schedules (beta^2 + gamma^2 = 1 for OneSidedSBDM);
* didt (autodiff of __call__) equals a central finite difference of __call__, with distinct
  x0, x1, z so a swapped argument or a missing noise derivative shows up.
"""
import pytest
import jax
import jax.numpy as jnp
from sinterp.interpolants import (
    interpolators, ConstantNoise, OneSidedLinear, OneSidedSBDM, LinearVPStochastic,
    TrigonometricVPStochastic, TrigonometricStochastic,
)

ONE_SIDED = (OneSidedLinear, OneSidedSBDM)


def _xyz(shape=(3, 2)):
    k = jax.random.split(jax.random.PRNGKey(0), 3)
    return (jax.random.normal(k[0], shape), jax.random.normal(k[1], shape) + 3.0,
            jax.random.normal(k[2], shape) - 1.0)


@pytest.mark.parametrize("interp_class", interpolators)
def test_endpoints(interp_class):
    i = interp_class()
    x0, x1, z = _xyz()
    if isinstance(i, ConstantNoise):
        start, end = x0 + i.sigma * z, x1 + i.sigma * z
        assert jnp.allclose(i.gamma(jnp.linspace(0, 1, 7)), i.sigma)
    elif isinstance(i, ONE_SIDED):
        start, end = z, x1
    else:
        start, end = x0, x1
    assert jnp.allclose(i(x0, x1, z, 0.0), start, atol=1e-6)
    assert jnp.allclose(i(x0, x1, z, 1.0), end, atol=1e-6)


@pytest.mark.parametrize("interp_class", [LinearVPStochastic, TrigonometricVPStochastic,
                                          TrigonometricStochastic, OneSidedSBDM])
def test_variance_preserving(interp_class):
    i = interp_class()
    t = jnp.linspace(0.0, 1.0, 101)
    total = i.alpha(t) ** 2 + i.beta(t) ** 2 + i.gamma(t) ** 2
    assert jnp.allclose(total, 1.0, atol=1e-6)


@pytest.mark.parametrize("interp_class", interpolators)
@pytest.mark.parametrize("t", [0.13, 0.37, 0.71, 0.92])   # away from the t=0.5 kinks
def test_didt_matches_finite_difference(interp_class, t):
    i = interp_class()
    x0, x1, z = _xyz((4, 3, 1))        # image-like trailing dims as well
    h = 1e-3
    fd = (i(x0, x1, z, t + h) - i(x0, x1, z, t - h)) / (2 * h)
    d = i.didt(x0, x1, z, t)
    assert d.shape == x0.shape
    assert jnp.allclose(d, fd, rtol=2e-3, atol=2e-3)
    # the deterministic part drops exactly the gamma' z term
    dd = i.didt_deterministic(x0, x1, t)
    assert jnp.allclose(d - dd, i.dgammadt(t) * z, atol=1e-5)
