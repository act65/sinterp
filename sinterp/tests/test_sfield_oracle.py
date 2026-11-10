"""Claim-driven tests for the stochastic-field parameterisations and their losses.

Fixture: x0 ~ N(m0, C0), x1 ~ N(m1, C1), z ~ N(0, I) independent, in 2-D with correlated,
unequal covariances and non-zero means (an isotropic zero-mean fixture would hide swapped
alpha/beta terms and sign errors). For this fixture every conditional expectation is a
closed-form linear regression, derived here from the joint Gaussian law of (x0, x1, z, x_t):

    Sigma_t = alpha^2 C0 + beta^2 C1 + gamma^2 I,   mu_t = alpha m0 + beta m1,   K = Sigma_t^-1
    eta0  = E[x0 | x_t] = m0 + alpha C0 K (x - mu_t)
    eta1  = E[x1 | x_t] = m1 + beta  C1 K (x - mu_t)
    etaz  = E[z  | x_t] = gamma K (x - mu_t)
    s     = grad log p_t = -K (x - mu_t)
    v     = alpha' eta0 + beta' eta1,   b = v + gamma' etaz

(the code's convention, see sinterp/README.md: b is E[dI/dt | x_t], v its noise-free part.)
This is an independent derivation: the library's exact GMM module computes b from the
covariance derivative instead, and we check the two agree.

Claims tested:
1. Each parameterisation (v/etaz, b/s, x0/x1, b/eta1, one-sided b, one-sided eta1), given the
   exact quantities it learns, reconstructs *all six* quantities exactly.
2. The population minimiser of each loss is the exact conditional expectation: the
   derivative of the expected loss along any perturbation direction g(x_t) vanishes at the
   oracle (checked by Monte Carlo, tolerance 5 standard errors).
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from sinterp import exact_si_gmm
from sinterp.interpolants import TrigonometricVPStochastic, LinearVPStochastic, OneSidedLinear
from sinterp.stochasticfield import (
    TwoSidedField_v_etaz, TwoSidedField_b_s, TwoSidedField_x0_x1, TwoSidedField_b_eta1,
    OneSidedField_b, OneSidedField_eta1,
)

M0 = jnp.array([-1.0, 0.5])
C0 = jnp.array([[1.0, 0.3], [0.3, 0.5]])
M1 = jnp.array([2.0, -1.0])
C1 = jnp.array([[0.2, -0.05], [-0.05, 0.4]])
TS = [0.3, 0.7]


def exact_quantities(interp, one_sided=False):
    """Closed-form conditional expectations (see module docstring)."""
    m0 = jnp.zeros(2) if one_sided else M0
    c0 = jnp.zeros((2, 2)) if one_sided else C0

    def coeffs(t):
        a, b, g = interp.alpha(t), interp.beta(t), interp.gamma(t)
        da, db, dg = (jax.grad(f)(t) for f in (interp.alpha, interp.beta, interp.gamma))
        return a, b, g, da, db, dg

    def all_q(x, t):
        a, b, g, da, db, dg = coeffs(t)
        sig = a**2 * c0 + b**2 * C1 + g**2 * jnp.eye(2)
        r = jnp.linalg.solve(sig, x - (a * m0 + b * M1))
        eta0 = m0 + a * c0 @ r
        eta1 = M1 + b * C1 @ r
        etaz = g * r
        v = da * eta0 + db * eta1
        return dict(eta0=eta0, eta1=eta1, etaz=etaz, s=-r, v=v, b=v + dg * etaz)

    return {k: (lambda k: lambda x, t, *a: all_q(x, t)[k])(k)
            for k in ("eta0", "eta1", "etaz", "s", "v", "b")}


class Fn(eqx.Module):
    f: callable = eqx.field(static=True)

    def __call__(self, x, t, *args):
        return self.f(x, t)


def _getters(field):
    return dict(eta0=field.get_denoiser_eta0, eta1=field.get_denoiser_eta1,
                etaz=field.get_denoiser_etaz, s=field.get_score_s,
                v=field.get_velocity_v, b=field.get_velocity_b)


TWO_SIDED = [
    (TwoSidedField_v_etaz, ("v", "etaz")),
    (TwoSidedField_b_s, ("b", "s")),
    (TwoSidedField_x0_x1, ("eta0", "eta1")),
    (TwoSidedField_b_eta1, ("b", "eta1")),
]
ONE_SIDED = [(OneSidedField_b, ("b",)), (OneSidedField_eta1, ("eta1",))]


def _points(n=5):
    return 1.5 * jax.random.normal(jax.random.PRNGKey(7), (n, 2)) + jnp.array([0.5, -0.3])


@pytest.mark.parametrize("interp_cls", [TrigonometricVPStochastic, LinearVPStochastic])
@pytest.mark.parametrize("cls, learned", TWO_SIDED, ids=[c.__name__ for c, _ in TWO_SIDED])
@pytest.mark.parametrize("t", TS)
def test_two_sided_reconstructs_all_quantities_from_oracle(interp_cls, cls, learned, t):
    """Claim 1 (two-sided). Catches: wrong 2x2 solve (swapped alpha/beta, wrong det sign),
    wrong sign in s = -etaz/gamma or v = b + gamma*gamma'*s, etaz formula errors."""
    interp = interp_cls()
    ex = exact_quantities(interp)
    field = cls(interp, *(Fn(ex[k]) for k in learned))
    for x in _points():
        for name, get in _getters(field).items():
            got, want = get(x, t), ex[name](x, t)
            assert jnp.allclose(got, want, rtol=1e-4, atol=1e-4), (cls.__name__, name, got, want)


@pytest.mark.parametrize("cls, learned", ONE_SIDED, ids=[c.__name__ for c, _ in ONE_SIDED])
@pytest.mark.parametrize("t", TS)
def test_one_sided_reconstructs_all_quantities_from_oracle(cls, learned, t):
    """Claim 1 (one-sided, x_t = (1-t) z + t x1). eta0 is identically 0 by convention."""
    interp = OneSidedLinear()
    ex = exact_quantities(interp, one_sided=True)
    field = cls(interp, Fn(ex[learned[0]]))
    for x in _points():
        for name, get in _getters(field).items():
            want = jnp.zeros(2) if name == "eta0" else ex[name](x, t)
            assert jnp.allclose(get(x, t), want, rtol=1e-4, atol=1e-4), (cls.__name__, name)


@pytest.mark.parametrize("one_sided", [False, True])
@pytest.mark.parametrize("t", TS)
def test_closed_form_oracle_matches_exact_gmm_module(one_sided, t):
    """The denoiser-based b and s above agree with sinterp.exact_si_gmm (derived from the
    covariance derivative). Guards the oracle itself against a shared mistake."""
    interp = OneSidedLinear() if one_sided else TrigonometricVPStochastic()
    ex = exact_quantities(interp, one_sided)
    m0, c0 = (jnp.zeros(2), jnp.zeros((2, 2))) if one_sided else (M0, C0)
    _, b, s = exact_si_gmm.construct_p_b_s(interp, [m0], [M1], [c0], [C1], jnp.ones(1), jnp.ones(1))
    for x in _points():
        assert jnp.allclose(b(x, t), ex["b"](x, t), atol=1e-4)
        assert jnp.allclose(s(x, t), ex["s"](x, t), atol=1e-4)


# --- Claim 2: population optimum of each loss -------------------------------------------

def _sample(key, n, one_sided=False):
    k0, k1, kz = jax.random.split(key, 3)
    x0 = M0 + jax.random.normal(k0, (n, 2)) @ jnp.linalg.cholesky(C0).T
    x1 = M1 + jax.random.normal(k1, (n, 2)) @ jnp.linalg.cholesky(C1).T
    if one_sided:
        x0 = jnp.zeros_like(x0)
    return x0, x1, jax.random.normal(kz, (n, 2))


def _g(x):
    """Perturbation direction: nonlinear so it is not orthogonal to errors by accident."""
    return jnp.tanh(x) + 0.3 * x[::-1] ** 2


# (field class, learned quantities in constructor order, loss name, which quantity is perturbed)
LOSS_CASES = [
    (TwoSidedField_b_s, ("b", "s"), "loss_b", "b"),
    (TwoSidedField_v_etaz, ("v", "etaz"), "loss_v", "v"),
    (TwoSidedField_v_etaz, ("v", "etaz"), "loss_etaz", "etaz"),
    (TwoSidedField_x0_x1, ("eta0", "eta1"), "loss_eta0", "eta0"),
    (TwoSidedField_x0_x1, ("eta0", "eta1"), "loss_eta1", "eta1"),
    (TwoSidedField_b_s, ("b", "s"), "loss_s", "s"),
]


def _loss_derivative_z(cls, learned, loss_name, perturbed, t, one_sided=False, n=20000):
    interp = OneSidedLinear() if one_sided else TrigonometricVPStochastic()
    ex = exact_quantities(interp, one_sided)
    x0, x1, z = _sample(jax.random.PRNGKey(11), n, one_sided)

    def per_sample_loss(eps, a, b, e):
        models = [Fn((lambda k: (lambda x, tt: ex[k](x, tt) + eps * _g(x)) if k == perturbed
                      else ex[k])(k)) for k in learned]
        return getattr(cls(interp, *models), loss_name)(a, b, e, t)

    d = jax.jit(jax.vmap(jax.grad(per_sample_loss), in_axes=(None, 0, 0, 0)))(0.0, x0, x1, z)
    return float(jnp.mean(d) / (jnp.std(d) / jnp.sqrt(n))), float(jnp.mean(d))


@pytest.mark.parametrize("cls, learned, loss_name, perturbed", LOSS_CASES)
@pytest.mark.parametrize("t", TS)
def test_loss_is_stationary_at_conditional_expectation(cls, learned, loss_name, perturbed, t):
    """Claim 2. d/d eps E[loss(oracle + eps g)] at eps=0 must be 0 (|z-score| < 5).
    Catches a loss regressing on the wrong target (e.g. the deterministic instead of the
    stochastic time derivative, x1 instead of x0, missing noise term)."""
    zscore, mean = _loss_derivative_z(cls, learned, loss_name, perturbed, t)
    assert abs(zscore) < 5, f"{loss_name}: mean dL/deps = {mean:.4g} ({zscore:.1f} s.e.)"


@pytest.mark.parametrize("cls, learned, loss_name, perturbed", [
    (OneSidedField_b, ("b",), "loss_b", "b"),
    (OneSidedField_eta1, ("eta1",), "loss_eta1", "eta1"),
])
@pytest.mark.parametrize("t", TS)
def test_one_sided_loss_is_stationary_at_conditional_expectation(cls, learned, loss_name, perturbed, t):
    zscore, mean = _loss_derivative_z(cls, learned, loss_name, perturbed, t, one_sided=True)
    assert abs(zscore) < 5, f"{loss_name}: mean dL/deps = {mean:.4g} ({zscore:.1f} s.e.)"


@pytest.mark.parametrize("wrong", ["v_for_b", "b_plus_0.1"])
@pytest.mark.parametrize("t", TS)
def test_stationarity_check_detects_wrong_oracle(wrong, t):
    """Power check for the stationarity test: regressing b on the noise-free velocity v
    (dropping gamma' * etaz) or a 0.1 offset must be rejected (observed |z| >= 6)."""
    interp = TrigonometricVPStochastic()
    ex = exact_quantities(interp)
    bad = ex["v"] if wrong == "v_for_b" else (lambda x, tt: ex["b"](x, tt) + 0.1)
    x0, x1, z = _sample(jax.random.PRNGKey(11), 20000)

    def per_sample_loss(eps, a, b, e):
        field = TwoSidedField_b_s(interp, Fn(lambda x, tt: bad(x, tt) + eps * _g(x)), Fn(ex["s"]))
        return field.loss_b(a, b, e, t)

    d = jax.jit(jax.vmap(jax.grad(per_sample_loss), in_axes=(None, 0, 0, 0)))(0.0, x0, x1, z)
    assert abs(jnp.mean(d) / (jnp.std(d) / jnp.sqrt(d.size))) > 5
