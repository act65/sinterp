"""Claim-driven tests for flow (integrators, Flow, Process, divergence estimators).

Imports only flow, sinterp, numpy/jax/equinox (this file ships in the release, unlike
test_flow.py / test_integration.py).

Claims:
1. Euler / RK4 converge at order 1 / 4 (observed order from step halving, float64) on a
   time-dependent linear ODE with closed-form solution.
2. Flow.backward inverts Flow.forward for a time-dependent, nonlinear field (catches using
   v(x, t) instead of -v(x, 1 - t) in the reversed ODE, which a time-independent field hides).
3. push_forward_log_prob / push_backward_log_prob match the analytic change of variables
   log p1(T(x0)) = log p0(x0) - log|det dT/dx0| for a *nonlinear* flow whose divergence varies
   along the path (dx/dt = sin(x): T has a closed form), also for dict (PyTree) states.
4. The Hutchinson estimator is unbiased for tr(J) and has the Gaussian-probe variance
   2 ||(J + J^T)/2||_F^2, for a dense non-symmetric J.
5. Euler-Maruyama adds diffusion * sqrt(dt) * N(0, 1) per step, independently per leaf.
6. Process: the forward / backward SDEs with drift b +- eps s and diffusion sqrt(2 eps)
   transport source to target (and back) with the exact Gaussian interpolant fields.
"""
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pytest

from flow.flow import Flow
from flow.process import Process
from flow.integrators import Euler, RK4, EulerMaruyama
from flow.utils import divergence_jvp, divergence_hutchinson


class Fn(eqx.Module):
    f: callable = eqx.field(static=True)

    def __call__(self, x, t, *args):
        return self.f(x, t)


# --- 1. convergence order --------------------------------------------------------------

M = np.array([[-0.5, 1.0], [-1.0, 0.2]])


def _linear_ode(x, t):
    return jnp.cos(3.0 * t) * (jnp.asarray(M, x.dtype) @ x)


def _linear_exact(x0):
    # x(1) = expm(M * int_0^1 cos(3t) dt) x0 (the matrices at different t commute)
    import scipy.linalg
    return scipy.linalg.expm(M * np.sin(3.0) / 3.0) @ np.asarray(x0)


@pytest.mark.parametrize("cls, order", [(Euler, 1), (RK4, 4)])
def test_integrator_convergence_order(cls, order):
    """Observed order log2(err(n) / err(2n)) must be within 0.15 of the theoretical order.
    Catches e.g. RK4 stage times off by dt/2 (drops to order 1-2 on a time-dependent field)."""
    with jax.enable_x64(True):
        x0 = jnp.array([1.0, -0.5], dtype=jnp.float64)
        exact = _linear_exact(x0)
        ns = [16, 32, 64] if order == 4 else [32, 64, 128, 256]
        solve = lambda n: cls(_linear_ode).solve(x0, 0.0, 1.0, n, None)
        errs = [np.linalg.norm(np.asarray(solve(n)) - exact) for n in ns]
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(np.abs(rates - order) < 0.15), (errs, rates)


def test_trajectory_endpoints_and_grid():
    """trajectory has n+1 states, starts at x0, ends at solve(), and state k is the solution
    at t = k/n (checked against the closed form with RK4 at small error)."""
    with jax.enable_x64(True):
        x0 = jnp.array([1.0, -0.5], dtype=jnp.float64)
        n = 40
        rk = RK4(_linear_ode)
        traj = rk.trajectory(x0, 0.0, 1.0, n, None)
        assert traj.shape == (n + 1, 2)
        assert jnp.array_equal(traj[0], x0)
        assert jnp.allclose(traj[-1], rk.solve(x0, 0.0, 1.0, n, None), atol=1e-12)
        import scipy.linalg
        k = 17
        want = scipy.linalg.expm(M * np.sin(3.0 * k / n) / 3.0) @ np.asarray(x0)
        assert np.allclose(np.asarray(traj[k]), want, atol=1e-6)


# --- 2. backward inverts forward ------------------------------------------------------

@pytest.mark.parametrize("integrator, atol", [("rk4", 1e-5), ("euler", 2e-2)])
def test_backward_inverts_forward_time_dependent(integrator, atol):
    v = Fn(lambda x, t: jnp.sin(x) * jnp.cos(4.0 * t) + jnp.array([1.0, -2.0]) * t ** 2)
    flow = Flow(v, n_steps=100, integrator_name=integrator)
    x = jnp.array([0.3, -1.1])
    assert jnp.allclose(flow.backward(flow.forward(x)), x, atol=atol)
    # and the reversed path really is the forward path read backwards
    fwd = flow.forward_trajectory(x)
    bwd = flow.backward_trajectory(fwd[-1])
    assert jnp.allclose(jnp.flip(bwd, 0), fwd, atol=atol)


# --- 3. push-forward log density ------------------------------------------------------

def _sin_flow_map(x0):
    """Exact time-1 map of dx/dt = sin(x): tan(x/2) = tan(x0/2) e^t (for |x0| < pi)."""
    return 2.0 * jnp.arctan(jnp.tan(x0 / 2.0) * jnp.e)


def _analytic_logdet(x0):
    return jnp.sum(jnp.log(jax.vmap(jax.grad(_sin_flow_map))(x0)))


def test_push_forward_log_prob_matches_change_of_variables():
    """log p1(x1) = log p0(x0) - log|det dT/dx0| with T the exact flow map. dx/dt = sin(x) has
    divergence f(t) = sum cos(x(t)) varying along the path (with a linear field any quadrature
    of the constant divergence is exact, which hides grid errors).

    The divergence is accumulated with a LEFT Riemann sum on the trajectory grid, so the
    log-density carries the known first-order error (f(1) - f(0)) / (2n) even though the
    state is integrated with RK4. We check the result against truth + that leading term to
    1e-4: this pins the grid (a right-endpoint or shifted grid flips/changes the term) and the
    sign of the divergence. Backward: the reversed grid evaluates the other endpoints, so the
    round trip log p0 -> log p1 -> log p0 is off by (f(1) - f(0)) / n."""
    n = 400
    flow = Flow(Fn(lambda x, t: jnp.sin(x)), n_steps=n, integrator_name="rk4")
    x0 = jnp.array([0.4, -1.3, 2.0])
    logp0 = jnp.array(-1.7)
    logp1, x1 = flow.push_forward_log_prob(logp0, x0)
    assert jnp.allclose(x1, _sin_flow_map(x0), atol=1e-5)
    lead = (jnp.sum(jnp.cos(x1)) - jnp.sum(jnp.cos(x0))) / (2 * n)
    truth = logp0 - _analytic_logdet(x0)
    assert abs(lead) > 1e-3                       # the fixture makes the term visible
    assert jnp.allclose(logp1, truth + lead, atol=1e-4), (logp1 - truth, lead)
    logp0_back, x0_back = flow.push_backward_log_prob(logp1, x1)
    assert jnp.allclose(x0_back, x0, atol=1e-5)
    assert jnp.allclose(logp0_back, logp0 + 2 * lead, atol=1e-4), (logp0_back - logp0, 2 * lead)


def test_push_forward_log_prob_pytree_state():
    """A dict state is flattened for the divergence: the result equals the array case."""
    flow = Flow(Fn(lambda x, t: jax.tree_util.tree_map(jnp.sin, x)), n_steps=50)
    flat_flow = Flow(Fn(lambda x, t: jnp.sin(x)), n_steps=50)
    x0 = {"a": jnp.array([0.4]), "b": jnp.array([-1.3, 2.0])}
    flat = jnp.array([0.4, -1.3, 2.0])
    logp1, x1 = flow.push_forward_log_prob(0.0, x0)
    want, x1_flat = flat_flow.push_forward_log_prob(0.0, flat)
    assert jnp.allclose(logp1, want, atol=1e-6)
    assert jnp.allclose(x1["b"], x1_flat[1:], atol=1e-6)


def test_batched_push_forward_with_hutchinson_is_unbiased():
    """b_push_forward_log_prob with a key uses the Hutchinson estimator; averaged over keys it
    must agree with the exact-divergence result (5 s.e.), and individual draws must differ
    (each sample gets its own key)."""
    A = jnp.array([[0.3, 1.0, 0.0], [-0.5, -0.2, 0.4], [0.1, 0.0, 0.1]])
    flow = Flow(Fn(lambda x, t: jnp.tanh(A @ x) + 0.1 * t), n_steps=50)
    x0 = jnp.array([0.4, -1.3, 2.0])
    reps = 2000
    exact, _ = flow.push_forward_log_prob(0.0, x0)
    est, _ = flow.b_push_forward_log_prob(jnp.zeros(reps), jnp.tile(x0, (reps, 1)), jax.random.PRNGKey(3))
    assert jnp.std(est) > 0
    assert abs(jnp.mean(est) - exact) < 5 * jnp.std(est) / np.sqrt(reps), (jnp.mean(est), exact)


# --- 4. divergence estimators ----------------------------------------------------------

def test_hutchinson_unbiased_with_gaussian_probe_variance():
    J = jnp.array([[1.0, 2.0, -1.0, 0.0], [0.5, -3.0, 0.0, 1.0],
                   [0.0, 1.5, 2.0, -0.5], [1.0, 0.0, 0.3, 0.7]])
    f = lambda x: J @ x
    x = jnp.array([0.1, 0.2, 0.3, 0.4])
    est = divergence_hutchinson(f)
    n = 20000
    draws = jax.jit(jax.vmap(lambda k: est(x, key=k)))(jax.random.split(jax.random.PRNGKey(0), n))
    se = jnp.std(draws) / np.sqrt(n)
    assert abs(jnp.mean(draws) - jnp.trace(J)) < 5 * se
    S = (J + J.T) / 2
    var_theory = 2 * jnp.sum(S ** 2)   # Gaussian probes; Rademacher would give 2 sum_{i!=j} S_ij^2
    # se of the sample variance ~ var * sqrt(2 / n) * sqrt(1 + kurtosis/2); 10% is > 5 se here
    assert abs(jnp.var(draws) / var_theory - 1) < 0.1, (jnp.var(draws), var_theory)
    # exact estimator on the same field
    assert jnp.allclose(divergence_jvp(f)(x), jnp.trace(J), atol=1e-5)


def test_hutchinson_step_idx_changes_probe():
    """Different integration steps must use different probes (fold_in(key, step_idx))."""
    f = lambda x: x ** 2
    x = jnp.arange(1.0, 6.0)
    est = divergence_hutchinson(f)
    k = jax.random.PRNGKey(0)
    assert est(x, key=k, step_idx=0) != est(x, key=k, step_idx=1)
    assert est(x, key=k, step_idx=3) == est(x, key=k, step_idx=3)


def test_divergence_jvp_dense_jacobian_pytree():
    """Exact divergence of a dense nonlinear map over a dict input equals the trace of the
    autodiff Jacobian of the flattened map."""
    W = jax.random.normal(jax.random.PRNGKey(1), (5, 5))

    def f(d):
        flat = jnp.concatenate([d["a"], d["b"].ravel()])
        out = jnp.tanh(W @ flat)
        return {"a": out[:1], "b": out[1:].reshape(2, 2)}

    d = {"a": jnp.array([0.3]), "b": jnp.array([[0.1, -0.4], [0.9, 0.2]])}
    flat = jnp.concatenate([d["a"], d["b"].ravel()])
    want = jnp.trace(jax.jacfwd(lambda v: jnp.tanh(W @ v))(flat))
    assert jnp.allclose(divergence_jvp(f)(d), want, atol=1e-5)


# --- 5. Euler-Maruyama noise -----------------------------------------------------------

def test_euler_maruyama_increment_law_and_leaf_independence():
    """Zero drift, diffusion sigma: x(1) - x(0) ~ N(0, sigma^2) for every leaf, and leaves get
    independent noise. Tolerances: 5 s.e. for the mean and correlation (se = 1/sqrt(n)), and
    5 s.e. of the variance ratio (sqrt(2/n))."""
    sigma, n = 0.7, 4000
    em = EulerMaruyama(lambda x, t: jax.tree_util.tree_map(jnp.zeros_like, x), lambda t: sigma)
    x0 = {"a": jnp.zeros(1), "b": jnp.zeros(1)}
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    out = jax.jit(jax.vmap(lambda k: em.solve(x0, 0.0, 1.0, 10, k)))(keys)
    a, b = np.asarray(out["a"]).ravel(), np.asarray(out["b"]).ravel()
    for v in (a, b):
        assert abs(v.mean()) < 5 * sigma / np.sqrt(n)
        assert abs(v.var() / sigma ** 2 - 1) < 5 * np.sqrt(2 / n)
    assert abs(np.corrcoef(a, b)[0, 1]) < 5 / np.sqrt(n)


# --- 6. Process (SDE) end to end -------------------------------------------------------

def test_process_forward_and_backward_transport_gaussians():
    """With the exact interpolant fields b, s for N(m0, c0) -> N(m1, c1) (1-D) and eps = 0.5,
    the forward SDE maps source samples to the target law and the backward SDE maps target
    samples to the source law. Tolerance: 5 MC s.e. plus Euler-Maruyama bias O(dt) (200 steps;
    observed bias < 0.02 in mean and variance)."""
    from sinterp import exact_si_gmm
    from sinterp.interpolants import TrigonometricVPStochastic
    m0, c0, m1, c1 = -1.0, 0.5, 2.0, 0.3
    _, b, s = exact_si_gmm.construct_p_b_s(
        TrigonometricVPStochastic(sigma=0.5), [jnp.array([m0])], [jnp.array([m1])],
        [jnp.array([[c0]])], [jnp.array([[c1]])], jnp.ones(1), jnp.ones(1))
    proc = Process(b, s, lambda t: 0.5, n_steps=200)
    n = 4000
    k0, k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 4)
    x0 = m0 + np.sqrt(c0) * jax.random.normal(k0, (n, 1))
    x1 = m1 + np.sqrt(c1) * jax.random.normal(k1, (n, 1))
    fwd = np.asarray(proc.b_forward(x0, jax.random.split(k2, n))).ravel()
    bwd = np.asarray(proc.b_backward(x1, jax.random.split(k3, n))).ravel()
    for out, m, c in ((fwd, m1, c1), (bwd, m0, c0)):
        assert abs(out.mean() - m) < 5 * np.sqrt(c / n) + 0.02, (out.mean(), m)
        assert abs(out.var() / c - 1) < 5 * np.sqrt(2 / n) + 0.05, (out.var(), c)
