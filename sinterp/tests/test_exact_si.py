"""Exact reference fields (sinterp.exact_si_gmm, exact_si_gaussian, exact_ot_gmm).

These modules are the oracles other tests and experiments trust, so they are checked
against derivations made another way:
  * endpoints: p(., 0) / p(., 1) equal the source / target mixtures;
  * normalisation: p(., t) integrates to 1 (2-D quadrature);
  * score: s = grad_x log p (autodiff);
  * velocity: the continuity equation dp/dt + div(p b) = 0 (autodiff in x and t), plus a
    Monte Carlo check that b is the conditional expectation E[dI/dt | x_t].

Fixture: 2-D, 2 source x 3 target components with unequal weights, non-zero means and
correlated, non-commuting covariances, under a smooth *stochastic* interpolant
(TrigonometricVPStochastic, gamma > 0 inside (0, 1)). Unequal n != m and unequal weights
catch a transposed weight matrix; gamma > 0 catches a dropped noise-covariance term.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from jax.scipy.stats import multivariate_normal as mvn

from sinterp import exact_si_gmm, exact_si_gaussian, exact_ot_gmm
from sinterp.interpolants import LinearDeterministic, TrigonometricVPStochastic

M0S = [jnp.array([-1.0, 0.5]), jnp.array([1.5, -0.5])]
C0S = [jnp.array([[1.0, 0.3], [0.3, 0.5]]), jnp.array([[0.4, -0.1], [-0.1, 0.8]])]
W0 = jnp.array([0.3, 0.7])
M1S = [jnp.array([2.0, 1.0]), jnp.array([-2.0, 0.0]), jnp.array([0.0, -2.5])]
C1S = [jnp.array([[0.2, 0.05], [0.05, 0.3]]), jnp.array([[0.5, 0.0], [0.0, 0.1]]),
       jnp.array([[0.3, -0.2], [-0.2, 0.6]])]
W1 = jnp.array([0.2, 0.5, 0.3])

ZS = [jnp.array([0.3, -0.4]), jnp.array([-1.2, 0.8]), jnp.array([1.7, 0.2])]
TS = [0.2, 0.5, 0.83]


def mixture(z, ms, cs, ws):
    return sum(w * mvn.pdf(z, m, c) for w, m, c in zip(ws, ms, cs))


def si_gmm():
    return exact_si_gmm.construct_p_b_s(TrigonometricVPStochastic(sigma=0.5), M0S, M1S, C0S, C1S, W0, W1)


def ot_gmm():
    return exact_ot_gmm.construct_optimal_transport(
        jnp.stack(M0S), jnp.stack(M1S), jnp.stack(C0S), jnp.stack(C1S), W0, W1)


BUILDERS = {"si_gmm": si_gmm, "ot_gmm": ot_gmm}


@pytest.mark.parametrize("which", list(BUILDERS))
def test_endpoints_are_source_and_target_mixtures(which):
    p, _, _ = BUILDERS[which]()
    for z in ZS:
        # ot_gmm adds a 1e-6 Cholesky jitter and its Sinkhorn marginals hold to ~1e-6
        assert jnp.allclose(p(z, 0.0), mixture(z, M0S, C0S, W0), rtol=1e-4, atol=1e-7)
        assert jnp.allclose(p(z, 1.0), mixture(z, M1S, C1S, W1), rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("which", list(BUILDERS))
@pytest.mark.parametrize("t", [0.37, 0.8])
def test_density_integrates_to_one(which, t):
    p, _, _ = BUILDERS[which]()
    g = jnp.linspace(-8.0, 8.0, 241)
    Z = jnp.stack(jnp.meshgrid(g, g), -1).reshape(-1, 2)
    mass = jnp.sum(jax.jit(jax.vmap(lambda z: p(z, t)))(Z)) * (g[1] - g[0]) ** 2
    assert abs(float(mass) - 1.0) < 1e-3


@pytest.mark.parametrize("which", list(BUILDERS))
@pytest.mark.parametrize("t", TS)
def test_score_is_grad_log_density(which, t):
    p, _, s = BUILDERS[which]()
    for z in ZS:
        want = jax.grad(lambda z: jnp.log(p(z, t)))(z)
        assert jnp.allclose(s(z, t), want, rtol=1e-3, atol=1e-4), (z, s(z, t), want)


@pytest.mark.parametrize("which", list(BUILDERS))
@pytest.mark.parametrize("t", TS)
def test_velocity_satisfies_continuity_equation(which, t):
    """dp/dt + div(p b) = 0. Catches a wrong b (missing factor 1/2 on dC, wrong component
    weights, sign error) that would leave p and s correct. Tolerance is relative to the size
    of the two terms."""
    p, b, _ = BUILDERS[which]()
    terms = jax.jit(lambda z: (jax.grad(p, argnums=1)(z, t),
                               jnp.trace(jax.jacfwd(lambda z: p(z, t) * b(z, t))(z))))
    for z in ZS:
        dpdt, div = terms(z)
        scale = jnp.abs(dpdt) + jnp.abs(div)
        assert jnp.abs(dpdt + div) <= 2e-3 * scale + 1e-7, (z, dpdt, div)


def test_gmm_velocity_matches_conditional_expectation_by_monte_carlo():
    """b(x, t) = E[dI/dt | x_t = x] (the target of the flow-matching loss). Continuity fixes b
    only up to p-weighted divergence-free fields, so check the regression directly:
    E[(dI/dt - b(x_t)) h(x_t)] = 0 for several test functions h, within 5 Monte Carlo s.e.
    (1-D, bimodal source, so the mixture posterior weights matter)."""
    interp = TrigonometricVPStochastic(sigma=0.5)
    m0s, c0s, w0 = [jnp.array([-1.0]), jnp.array([2.0])], [jnp.array([[0.3]]), jnp.array([[0.5]])], jnp.array([0.4, 0.6])
    m1s, c1s, w1 = [jnp.array([1.0])], [jnp.array([[0.2]])], jnp.array([1.0])
    _, b, _ = exact_si_gmm.construct_p_b_s(interp, m0s, m1s, c0s, c1s, w0, w1)
    n, t = 40000, 0.4
    k = random.split(random.PRNGKey(0), 4)
    comp = random.bernoulli(k[0], 0.6, (n,)).astype(int)
    x0 = jnp.array([-1.0, 2.0])[comp] + jnp.sqrt(jnp.array([0.3, 0.5]))[comp] * random.normal(k[1], (n,))
    x1 = 1.0 + jnp.sqrt(0.2) * random.normal(k[2], (n,))
    z = random.normal(k[3], (n,))
    xt = interp(x0, x1, z, t)
    dI = (interp.dalphadt(t) * x0 + interp.dbetadt(t) * x1 + interp.dgammadt(t) * z)
    resid = dI - jax.jit(jax.vmap(lambda x: b(x[None], t)[0]))(xt)
    for h in (jnp.ones_like, lambda x: x, jnp.tanh, lambda x: x**2):
        r = resid * h(xt)
        assert abs(jnp.mean(r)) < 5 * jnp.std(r) / jnp.sqrt(n)


def test_gmm_fields_stay_finite_far_from_every_component():
    """b and s weight the components by their posterior responsibilities. Computed as ratios
    of raw densities these underflow to 0/0 = nan in float32 a few dozen standard deviations
    from every component; computed from log-densities they stay finite, and far out the
    score tends to the dominant component's -C^{-1}(z - m). Catches a regression to the
    raw-density ratio."""
    p, b, s = si_gmm()
    for z in [jnp.array([40.0, -35.0]), jnp.array([-60.0, 10.0])]:
        for t in (0.2, 0.8):
            bz, sz = b(z, t), s(z, t)
            assert jnp.all(jnp.isfinite(bz)) and jnp.all(jnp.isfinite(sz)), (z, t, bz, sz)
            # agrees with autodiff of the log of the (log-space) mixture density
            logp = lambda x: jax.scipy.special.logsumexp(jnp.array(
                [jnp.log(w0 * w1) + mvn.logpdf(x, *_component(t, i, j))
                 for i, w0 in enumerate(W0) for j, w1 in enumerate(W1)]))
            assert jnp.allclose(sz, jax.grad(logp)(z), rtol=1e-3, atol=1e-3)


def _component(t, i, j):
    interp = TrigonometricVPStochastic(sigma=0.5)
    m = interp(M0S[i], M1S[j], jnp.zeros(2), t)
    c = interp.alpha(t) ** 2 * C0S[i] + interp.beta(t) ** 2 * C1S[j] + interp.gamma(t) ** 2 * jnp.eye(2)
    return m, c


def test_gaussian_module_equals_gmm_module_for_deterministic_interpolant():
    """exact_si_gaussian (single Gaussian, gamma = 0) agrees with exact_si_gmm with one mode."""
    key = random.PRNGKey(0)
    m1, m2 = jnp.array([0.0, 1.0]), jnp.array([0.5, -1.0])
    c1, c2 = C0S[0], C1S[2]
    interp = LinearDeterministic()
    p0, v0, s0 = exact_si_gaussian.construct_p_b_s(m1, m2, c1, c2, interp)
    p1, v1, s1 = exact_si_gmm.construct_p_b_s(interp, [m1], [m2], [c1], [c2], jnp.ones(1), jnp.ones(1))
    for _ in range(10):
        key, k1, k2 = random.split(key, 3)
        z = random.uniform(k1, (2,)) * 6 - 3
        t = random.uniform(k2, ())
        assert jnp.isclose(p0(z, t), p1(z, t), rtol=1e-4)
        assert jnp.allclose(v0(z, t), v1(z, t), rtol=1e-4, atol=1e-5)
        assert jnp.allclose(s0(z, t), s1(z, t), rtol=1e-4, atol=1e-5)


def test_gaussian_module_handles_stochastic_interpolant():
    interp = TrigonometricVPStochastic(sigma=0.5)
    pg, vg, sg = exact_si_gaussian.construct_p_b_s(M0S[0], M1S[0], C0S[0], C1S[0], interp)
    p1, b1, s1 = exact_si_gmm.construct_p_b_s(interp, M0S[:1], M1S[:1], C0S[:1], C1S[:1], jnp.ones(1), jnp.ones(1))
    z, t = ZS[0], 0.4
    assert jnp.isclose(pg(z, t), p1(z, t), rtol=1e-4)
    assert jnp.allclose(vg(z, t), b1(z, t), rtol=1e-4)


def test_ot_gmm_coupling_has_mixture_weights_as_marginals():
    pi = exact_ot_gmm.get_ott_coupling(jnp.stack(M0S), jnp.stack(M1S), jnp.stack(C0S), jnp.stack(C1S), W0, W1)
    assert jnp.all(pi >= 0)
    assert jnp.allclose(pi.sum(1), W0, atol=1e-5)
    assert jnp.allclose(pi.sum(0), W1, atol=1e-5)


def test_bures_wasserstein_cost_closed_forms():
    """W2^2 between Gaussians with commuting (diagonal) covariances is
    ||m0 - m1||^2 + sum_k (sd0_k - sd1_k)^2. Catches a missing factor 2 or an unsquared sqrtm."""
    m0s, m1s = jnp.array([[0.0, 1.0]]), jnp.array([[2.0, -1.0], [0.5, 0.5]])
    c0s = jnp.array([jnp.diag(jnp.array([4.0, 0.25]))])
    c1s = jnp.array([jnp.diag(jnp.array([1.0, 1.0])), jnp.diag(jnp.array([9.0, 0.04]))])
    C = exact_ot_gmm.compute_wasserstein_cost_matrix(m0s, m1s, c0s, c1s)
    want = np.array([[4 + 4 + (2 - 1) ** 2 + (0.5 - 1) ** 2,
                      0.25 + 0.25 + (2 - 3) ** 2 + (0.5 - 0.2) ** 2]])
    assert np.allclose(np.asarray(C), want, rtol=1e-5)


def test_ot_gmm_single_pair_is_the_gaussian_displacement_interpolation():
    """For one source and one target Gaussian with non-commuting covariances, the path is
    McCann's interpolation: at t = 1 it is exactly N(m1, C1) (so the map matrix A satisfies
    A C0 A = C1), and its mean moves on the straight line (1 - t) m0 + t m1."""
    p, _, _ = exact_ot_gmm.construct_optimal_transport(
        jnp.stack(M0S[:1]), jnp.stack(M1S[2:]), jnp.stack(C0S[:1]), jnp.stack(C1S[2:]), jnp.ones(1), jnp.ones(1))
    for z in ZS:
        assert jnp.allclose(p(z, 1.0), mvn.pdf(z, M1S[2], C1S[2]), rtol=1e-4)
    t = 0.3
    mean_t = (1 - t) * M0S[0] + t * M1S[2]
    assert jnp.allclose(jax.grad(lambda z: jnp.log(p(z, t)))(mean_t), 0.0, atol=1e-4)
