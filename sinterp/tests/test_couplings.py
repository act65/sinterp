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
        
        # 1. Verify x1_out is a permutation of x1 (mass conservation)
        assert jnp.allclose(jnp.sort(x1_out, axis=0), jnp.sort(x1, axis=0), atol=0.5)
        
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
            _, ref, _, _, _ = cp(key, x[n], e[n], c0=cond)
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
