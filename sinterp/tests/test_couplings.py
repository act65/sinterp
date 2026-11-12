import itertools
import numpy as np
import pytest
import jax.numpy as jnp
import jax
from sinterp.couplings import IndependentCoupling, ExactOTCoupling, ExactConditionalCoupling, SinkhornCoupling

@pytest.fixture
def key():
    return jax.random.PRNGKey(0)

class TestIndependentCoupling:
    def test_independence(self, key):
        """
        Test that x1 is actually shuffled.
        """
        N = 100
        k1, k2, k3 = jax.random.split(key, 3)
        # Create distinct values so shuffling is detectable
        x0 = jnp.arange(N).reshape(-1, 1).astype(jnp.float32)
        x1 = x0.copy()
        
        coupling = IndependentCoupling()
        _, x1_out, _, _, _ = coupling(k3, x0, x1)
        
        # If shuffled, x1_out should NOT equal x1 (probability of identity is 1/N!)
        assert not jnp.all(x1_out == x1), "IndependentCoupling did not shuffle the data!"
        
        # Correlation should be low (close to 0)
        corr = jnp.corrcoef(x0.flatten(), x1_out.flatten())[0, 1]
        assert abs(corr) < 0.3, f"Correlation too high for independent coupling: {corr}"

    def test_pytree_support(self, key):
        N = 10
        x0 = {"a": jnp.zeros((N, 1)), "b": jnp.ones((N, 2))}
        x1 = {"a": jnp.arange(N).reshape(-1, 1).astype(jnp.float32), 
              "b": jnp.arange(N, 3*N).reshape(-1, 2).astype(jnp.float32)}
        
        coupling = IndependentCoupling()
        _, x1_out, _, _, _ = coupling(key, x0, x1)
        
        assert isinstance(x1_out, dict)
        assert x1_out["a"].shape == (N, 1)
        assert x1_out["b"].shape == (N, 2)
        # Check that the same permutation was applied to both leaves
        # We can find the permutation by looking at "a"
        perm = jnp.round(x1_out["a"]).astype(jnp.int32).flatten()
        expected_b = x1["b"][perm]
        assert jnp.allclose(x1_out["b"], expected_b)

class TestExactOTCoupling:
    def test_1d_sorting_exact(self, key):
        """
        Exact OT on 1D data is equivalent to sorting.
        """
        N = 20 # Smaller N for faster Sinkhorn convergence
        k1, k2, k3 = jax.random.split(key, 3)
        x0 = jax.random.normal(k1, (N, 1))
        x1 = jax.random.normal(k2, (N, 1)) + 5.0
        
        coupling = ExactOTCoupling(epsilon=1e-2, n_iters=2000)
        _, x1_out, _, _, _ = coupling(k3, x0, x1)
        
        # 1. Verify x1_out is (approximately) the sorted matching. The plan is entropic
        # (eps=1e-2), so near-tied neighbours are blurred on the scale of their spacing;
        # max observed error 0.15 on this batch, a wrong (e.g. reversed) matching is O(2).
        assert jnp.allclose(x1_out[jnp.argsort(x0[:, 0])], jnp.sort(x1, axis=0), atol=0.2)
        
        # 2. Verify Monotonicity
        sort_idx = jnp.argsort(x0, axis=0).flatten()
        x1_out_sorted_by_x0 = x1_out[sort_idx]
        
        diffs = jnp.diff(x1_out_sorted_by_x0, axis=0)
        assert jnp.all(diffs >= -1e-3), "Coupling is not monotonic"

    def test_extra_args(self, key):
        N = 10
        x0 = jnp.linspace(0, 1, N).reshape(-1, 1)
        x1 = jnp.linspace(1, 0, N).reshape(-1, 1) # Reversed
        extra = jnp.arange(N).reshape(-1, 1)
        
        coupling = ExactOTCoupling(epsilon=1e-4)
        _, x1_out, _, _, extra_out = coupling(key, x0, x1, extra_arg=extra)
        
        # In 1D, x0=[0...1] and x1=[1...0] should result in x1_out=[0...1] (reversing x1)
        assert jnp.allclose(x1_out, x0, atol=1e-2)
        
        # The same reverse permutation should be applied to extra
        assert jnp.allclose(extra_out["extra_arg"], jnp.flip(extra, axis=0), atol=1e-2)

class TestConditionalOTCoupling:
    def test_beta_influence_exact(self, key):
        """
        Test the trade-off between spatial and conditional cost.
        """
        x0 = jnp.array([[0.], [3.]])
        c0 = jnp.array([[0.], [1.]])
        
        x1 = jnp.array([[0.], [3.]])
        c1 = jnp.array([[1.], [0.]])
        
        # Case A: Low Beta (Should pick Identity / Spatial match)
        coupling_low = ExactConditionalCoupling(beta=1.0, epsilon=1e-4)
        _, x1_low, _, c1_low, _ = coupling_low(key, x0, x1, c0, c1)
        
        assert jnp.allclose(x1_low[0], 0.0, atol=1e-2)
        assert jnp.allclose(c1_low[0], 1.0, atol=1e-2)
        
        # Case B: High Beta (Should pick Swap / Conditional match)
        coupling_high = ExactConditionalCoupling(beta=100.0, epsilon=1e-4)
        _, x1_high, _, c1_high, _ = coupling_high(key, x0, x1, c0, c1)
        
        assert jnp.allclose(x1_high[0], 3.0, atol=1e-2)
        assert jnp.allclose(c1_high[0], 0.0, atol=1e-2)


class TestC2OTCoupling:
    """C2OT (Cheng & Schwing 2025): condition-aware hard OT."""

    @staticmethod
    def _gaussian_batch(key, n, rho=0.9):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        y = jax.random.normal(k1, (n, 1))
        th = rho * y + jnp.sqrt(1 - rho**2) * jax.random.normal(k2, (n, 1))
        return y, th, jax.random.normal(k3, (n, 1))

    def test_output_is_permutation_of_noise(self, key):
        from sinterp.couplings import C2OTCoupling
        y, th, e = self._gaussian_batch(key, 64)
        _, e_out, _, _, _ = C2OTCoupling()(key, th, e, c0=y)
        assert jnp.allclose(jnp.sort(e_out.ravel()), jnp.sort(e.ravel()))

    def test_w_zero_is_exact_ot(self, key):
        """With no condition term, 1-D hard OT is sorting."""
        from sinterp.couplings import C2OTCoupling
        _, th, e = self._gaussian_batch(key, 50)
        _, e_out, _, _, _ = C2OTCoupling(w=0.0)(key, th, e, c0=th)
        order = jnp.argsort(th.ravel())
        assert jnp.allclose(e_out.ravel()[order], jnp.sort(e.ravel()))

    def test_huge_w_is_identity_for_distinct_conditions(self, key):
        from sinterp.couplings import C2OTCoupling
        y, th, e = self._gaussian_batch(key, 40)
        _, e_out, _, _, _ = C2OTCoupling(w=1e8)(key, th, e, c0=y)
        assert jnp.allclose(e_out, e)

    def test_discrete_never_crosses_classes(self, key):
        from sinterp.couplings import C2OTCoupling
        n = 60
        c = (jnp.arange(n) % 3).reshape(-1, 1).astype(jnp.float32)
        x = c * 4.0 + jax.random.normal(key, (n, 1))
        e = jax.random.normal(jax.random.fold_in(key, 1), (n, 2))
        perm, _ = C2OTCoupling(cond_dist='discrete').matching(x, e[:, :1], c0=c)
        assert jnp.all(c.ravel()[perm] == c.ravel()), "noise moved to another class"
        # within a class it is still OT: every class's x sorted against its noise
        for k in range(3):
            idx = jnp.where(c.ravel() == k)[0]
            paired = e[perm[idx], 0]
            assert jnp.allclose(paired[jnp.argsort(x[idx, 0])], jnp.sort(paired))

    def test_adaptive_w_hits_target_ratio(self, key):
        import numpy as np
        from sinterp.couplings import C2OTCoupling
        y, th, e = self._gaussian_batch(key, 128)
        cp = C2OTCoupling(target_ratio=0.05)
        _, w = cp.matching(th, e, c0=y)
        D = np.asarray(cp._sqdist(th, e)); Cc = np.asarray(cp._sqdist(y, y))
        r = np.mean(D + float(w) * Cc <= np.diag(D)[:, None])
        assert float(w) > 0 and abs(r - 0.05) < 0.01

    def test_removes_condition_dependence(self, key):
        """Joint OT makes the theta-noise track y; C2OT keeps it ~independent and still straightens."""
        import numpy as np
        from sinterp.couplings import C2OTCoupling
        Y, E_ot, E_c2, TH = [], [], [], []
        for k in jax.random.split(key, 40):
            y, th, e = self._gaussian_batch(k, 128)
            x = jnp.concatenate([y, th], 1)
            ey = jax.random.normal(jax.random.fold_in(k, 9), (128, 1))
            _, ej, _, _, _ = C2OTCoupling(w=0.0)(k, x, jnp.concatenate([ey, e], 1))   # joint hard OT
            _, ec, _, _, _ = C2OTCoupling(target_ratio=0.05)(k, th, e, c0=y)
            Y.append(y); TH.append(th); E_ot.append(ej[:, 1:]); E_c2.append(ec)
        Y, TH, E_ot, E_c2 = (np.asarray(jnp.concatenate(a)).ravel() for a in (Y, TH, E_ot, E_c2))
        assert np.corrcoef(Y, E_ot)[0, 1] > 0.3
        assert abs(np.corrcoef(Y, E_c2)[0, 1]) < 0.05
        # still couples within condition: path length below independent (= 2); the unbiased
        # floor (exact conditional OT) is rho^2 + (1 - sqrt(1 - rho^2))^2 = 1.13
        assert 1.1 < np.mean((TH - E_c2) ** 2) < 1.5

    def test_jit_and_couple_blocks(self, key):
        from sinterp.couplings import C2OTCoupling
        y, th, e = self._gaussian_batch(key, 32)
        x = {"y": y, "theta": th}; ed = {"y": e, "theta": e[::-1]}
        f = jax.jit(lambda k, x, ed: C2OTCoupling().couple_blocks(k, x, ed, {"y": [], "theta": ["y"]}))
        out = f(key, x, ed)
        assert set(out) == {"y", "theta"} and out["theta"].shape == (32, 1)
        assert jnp.allclose(jnp.sort(out["theta"].ravel()), jnp.sort(ed["theta"].ravel()))

    @pytest.mark.parametrize("target", [0.01, 0.05, 0.2])
    @pytest.mark.parametrize("dc", [1, 4, 12])
    def test_closed_form_adaptive_w_matches_bisection(self, target, dc):
        """The O(B^2) closed form for the adaptive weight equals the bisection it replaced
        (to bisection precision), and meets the target ratio on the right side."""
        import numpy as np
        from sinterp.couplings import C2OTCoupling as C
        rng = np.random.default_rng(dc)
        for trial in range(5):
            x, e, c = rng.normal(size=(128, 4)), rng.normal(size=(128, 4)), rng.normal(size=(128, dc))
            D = np.asarray(C._sqdist(jnp.asarray(x), jnp.asarray(e)), np.float64)
            Cc = np.asarray(C._sqdist(jnp.asarray(c), jnp.asarray(c)), np.float64)
            w, w_ref = C._adaptive_w(D, Cc, target), C._adaptive_w_bisect(D, Cc, target)
            assert abs(w - w_ref) <= 1e-9 * max(w_ref, 1.0), (trial, w, w_ref)
            assert C._ratio(D, Cc, w) <= target
            if w > 0:
                assert C._ratio(D, Cc, w * (1 - 1e-9)) > target

    def test_unattainable_target_matches_within_conditions(self):
        """With few distinct condition values, equal-condition pairs alone exceed the target
        ratio, so no finite weight reaches it. The adaptive rule then returns w = inf and the
        matching is the exact within-condition optimum (the cond_dist='discrete' matching),
        not a ~1e18 weight that swamps the transport cost. Catches a regression to a huge
        finite w (precision loss in D) or to plain OT ignoring the condition."""
        import numpy as np
        from sinterp.couplings import C2OTCoupling as C
        rng = np.random.default_rng(0)
        x, e = rng.normal(size=(64, 2)), rng.normal(size=(64, 2))
        c = np.repeat(np.array([[0.0], [1.0]]), 32, axis=0)          # 2 condition values
        D = np.asarray(C._sqdist(jnp.asarray(x), jnp.asarray(e)), np.float64)
        Cc = np.asarray(C._sqdist(jnp.asarray(c), jnp.asarray(c)), np.float64)
        assert np.isinf(C._adaptive_w(D, Cc, 0.05))
        cols, w = C(target_ratio=0.05)._host_match(D, Cc, True)
        cols_disc, _ = C(cond_dist="discrete")._host_match(D, Cc, True)
        assert np.isinf(w) and np.array_equal(cols, cols_disc)
        assert np.all(c[cols, 0] == c[:, 0])                          # never crosses conditions

    def test_batched_couple_blocks_equals_per_block_matching(self, key):
        """couple_blocks matches all blocks in one host call; the result must be exactly the
        per-block C2OT matchings (block noise matched to block values, parents as condition)."""
        from sinterp.couplings import C2OTCoupling
        ks = jax.random.split(key, 6)
        a = jax.random.normal(ks[0], (128, 2))
        b = a @ jnp.ones((2, 3)) * 0.5 + jax.random.normal(ks[1], (128, 3))
        c = jnp.concatenate([a, b], 1)[:, :4] + jax.random.normal(ks[2], (128, 4))
        x = {"a": a, "b": b, "c": c}
        e = {n: jax.random.normal(k, v.shape) for (n, v), k in zip(x.items(), ks[3:])}
        deps = {"a": [], "b": ["a"], "c": ["a", "b"]}
        cp = C2OTCoupling(target_ratio=0.05)
        out = jax.jit(lambda x, e: cp.couple_blocks(key, x, e, deps))(x, e)
        for n in x:
            cond = {p: x[p] for p in deps[n]} or None
            # jitted reference: eager and jitted costs can differ in the last bit, which can
            # flip an exactly tied Hungarian/adaptive-w decision (see test_hard_coupling_under_vmap)
            ref = jax.jit(lambda xn, en, c: cp(key, xn, en, c0=c)[1])(x[n], e[n], cond)
            assert jnp.array_equal(out[n], ref), n

    def test_hard_coupling_under_vmap(self, key):
        """vmap over learner seeds (sweep seed_batch) must work through the host callback and
        give the same result as unbatched calls."""
        from sinterp.couplings import C2OTCoupling
        y, th, e = self._gaussian_batch(key, 32)
        cp = C2OTCoupling(target_ratio=0.05)
        f = lambda th, e: cp.couple_blocks(key, {"y": y, "theta": th}, {"y": e, "theta": e[::-1]},
                                           {"y": [], "theta": ["y"]})["theta"]
        TH, E = jnp.stack([th, -th]), jnp.stack([e, 2 * e])
        out = jax.jit(jax.vmap(f))(TH, E)
        # Compare against the *jitted* single call: eager and jitted costs can differ in the
        # last bit, and the closed-form C2OT weight sits exactly on a threshold, so eager vs
        # jit can legitimately pick a different (equally optimal) pairing on CPU.
        for i in range(2):
            assert jnp.array_equal(out[i], jax.jit(f)(TH[i], E[i]))


def _brute_force_assignment(cost):
    n = cost.shape[0]
    return np.array(min(itertools.permutations(range(n)),
                        key=lambda p: cost[np.arange(n), list(p)].sum()))


class TestCouplingClaims:
    """Claims checked against independent references (brute force, ott, statistics)."""

    def test_independent_coupling_keeps_x1_c1_extra_paired(self, key):
        """x1, c1 and every extra arg are permuted by the *same* permutation; x0, c0 untouched.
        Catches permuting c1/extra with a different key or forgetting to permute them."""
        n = 50
        x0 = jnp.arange(n, dtype=jnp.float32)[:, None]
        x1 = 10.0 * x0 + 1.0
        c0 = -x0
        c1 = 100.0 * x0
        x0o, x1o, c0o, c1o, ex = IndependentCoupling()(key, x0, x1, c0, c1, tag=x0 * 7.0)
        idx = np.asarray((x1o[:, 0] - 1.0) / 10.0).round().astype(int)
        assert sorted(idx) == list(range(n))                      # a permutation
        assert jnp.array_equal(x0o, x0) and jnp.array_equal(c0o, c0)
        assert jnp.allclose(c1o[:, 0], 100.0 * idx) and jnp.allclose(ex["tag"][:, 0], 7.0 * idx)

    @pytest.mark.parametrize("seed", range(4))
    def test_hard_ot_minimises_cost_vs_brute_force(self, seed):
        """C2OT with w=0 is exact hard OT: its matching attains the brute-force minimum of
        sum_i ||x0_i - x1_perm(i)||^2 over all 6! permutations (2-D, shifted target)."""
        from sinterp.couplings import C2OTCoupling
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
        x0 = jax.random.normal(k1, (6, 2))
        x1 = jax.random.normal(k2, (6, 2)) + jnp.array([1.0, 0.0])
        _, x1o, _, _, _ = jax.jit(C2OTCoupling(w=0.0).__call__)(k3, x0, x1)
        best = _brute_force_assignment(np.asarray(((x0[:, None] - x1[None]) ** 2).sum(-1)))
        assert jnp.array_equal(x1o, x1[best])

    @pytest.mark.parametrize("seed", [0, 2])
    def test_exact_ot_coupling_approaches_brute_force(self, seed):
        """ExactOTCoupling (Sinkhorn, eps=1e-2) returns the barycentric projection, which for
        batches whose best and second-best assignments differ in cost by > 0.38 (seeds 0, 2)
        is within 0.02 of the brute-force optimal assignment (entropic blur ~ exp(-gap/eps))."""
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
        x0 = jax.random.normal(k1, (6, 2))
        x1 = jax.random.normal(k2, (6, 2)) + jnp.array([1.0, 0.0])
        _, x1o, _, _, _ = ExactOTCoupling()(k3, x0, x1)
        best = _brute_force_assignment(np.asarray(((x0[:, None] - x1[None]) ** 2).sum(-1)))
        assert jnp.allclose(x1o, x1[best], atol=2e-2)

    @pytest.mark.skipif(
        tuple(int(v) for v in jax.__version__.split(".")[:2]) < (0, 9) and jax.default_backend() == "cpu",
        reason="jaxlib 0.6 on CPU aborts when this callback is compiled together with the cost "
               "computation (XLA 'Could not convert JSON string to proto'); passes with jax >= 0.9")
    @pytest.mark.parametrize("w", [0.5, 3.0])
    @pytest.mark.parametrize("seed", range(3))
    def test_c2ot_fixed_w_minimises_combined_cost(self, w, seed):
        """C2OT claim: the matching minimises ||x0_i - x1_j||^2 + w ||c_i - c_j||^2 exactly."""
        from sinterp.couplings import C2OTCoupling
        ks = jax.random.split(jax.random.PRNGKey(seed), 3)
        x, e, c = (jax.random.normal(ks[0], (6, 2)), jax.random.normal(ks[1], (6, 2)),
                   jax.random.normal(ks[2], (6, 1)))
        cost = np.asarray(((x[:, None] - e[None]) ** 2).sum(-1) + w * ((c[:, None] - c[None]) ** 2).sum(-1))
        perm, w_used = jax.jit(lambda x, e, c: C2OTCoupling(w=w).matching(x, e, c0=c))(x, e, c)
        assert float(w_used) == pytest.approx(w)
        assert np.array_equal(np.asarray(perm), _brute_force_assignment(cost))

    def test_fixed_sinkhorn_matches_ott(self, key):
        """SinkhornCoupling's default 'fixed' backend claims the same updates as ott's balanced
        lse-mode solver, so with the iteration count pinned (ott min=max=100) the plans agree up
        to float32 rounding (~1e-4 relative). Its row marginals are exactly uniform (f is
        updated last). (Unpinned, ott may stop early at its 1e-3 threshold, so the default
        SinkhornCoupling(backend='ott') differs at the 1e-3 level on some batches.)"""
        from ott.geometry import pointcloud
        from ott.problems.linear import linear_problem
        from ott.solvers.linear import sinkhorn
        k1, k2 = jax.random.split(key)
        x0 = jax.random.normal(k1, (64, 2))
        x1 = 0.5 * jax.random.normal(k2, (64, 2)) + 1.0
        P = SinkhornCoupling._fixed_sinkhorn_plan(x0, x1, 0.1, 100)
        assert jnp.allclose(P.sum(1), 1 / 64, rtol=1e-4)
        ref = sinkhorn.Sinkhorn(min_iterations=100, max_iterations=100)(
            linear_problem.LinearProblem(pointcloud.PointCloud(x0, x1, epsilon=0.1))).matrix
        assert float(jnp.max(jnp.abs(P - ref)) / jnp.max(ref)) < 1e-3

    def test_couple_blocks_root_is_hard_ot_and_child_noise_independent_of_parent(self):
        """Per-block coupling claims: (i) a block with no conditions gets plain hard OT, i.e. in
        1-D its noise is sorted against its values; (ii) a child's noise is (approximately)
        independent of its parent: corr(e_child, x_parent) = 0 within 5 s.e. (se = 1/sqrt(2560)),
        whereas joint hard OT makes it ~ rho; (iii) child and parent noise stay uncorrelated,
        so the coupled noise vector is still ~ N(0, I)."""
        from sinterp.couplings import C2OTCoupling
        rho, n = 0.8, 128
        cp = C2OTCoupling(target_ratio=0.01)
        f = jax.jit(lambda k, x, e: cp.couple_blocks(k, x, e, {"a": [], "b": ["a"]}))
        A, EA, EB = [], [], []
        for k in jax.random.split(jax.random.PRNGKey(0), 20):
            k1, k2, k3, k4 = jax.random.split(k, 4)
            a = jax.random.normal(k1, (n, 1))
            b = rho * a + np.sqrt(1 - rho**2) * jax.random.normal(k2, (n, 1))
            e = {"a": jax.random.normal(k3, (n, 1)), "b": jax.random.normal(k4, (n, 1))}
            out = f(k, {"a": a, "b": b}, e)
            assert jnp.array_equal(out["a"][jnp.argsort(a[:, 0])], jnp.sort(e["a"], axis=0))   # (i)
            assert jnp.array_equal(jnp.sort(out["b"], axis=0), jnp.sort(e["b"], axis=0))
            A.append(a); EA.append(out["a"]); EB.append(out["b"])
        A, EA, EB = (np.asarray(jnp.concatenate(v)).ravel() for v in (A, EA, EB))
        se = 1 / np.sqrt(A.size)
        assert abs(np.corrcoef(EB, A)[0, 1]) < 5 * se                                          # (ii)
        assert abs(np.corrcoef(EB, EA)[0, 1]) < 5 * se                                         # (iii)
