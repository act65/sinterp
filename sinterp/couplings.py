import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

class Coupling:
    def __call__(self, key, x0, x1, c0=None, c1=None, **extra_args):
        """
        Return a coupling between two distributions.
        If c0, c1 are provided, they are used for conditional OT with cost weight beta.
        Any extra_args (pytrees) are permuted by the same coupling.
        
        Returns:
            x0, x1_coupled, c0, c1_coupled, extra_args_coupled
        """
        raise NotImplementedError

    def _flatten_pytree(self, pytree):
        """Flattens a pytree and concatenates into a single (B, D) array."""
        if pytree is None:
            return None, None
        leaves, treedef = jax.tree_util.tree_flatten(pytree)
        # Assuming each leaf has shape (B, ...)
        batch_size = leaves[0].shape[0]
        flat_leaves = [leaf.reshape(batch_size, -1) for leaf in leaves]
        return jnp.concatenate(flat_leaves, axis=1), treedef

    def _apply_permutation(self, pytree, p_matrix):
        """
        Applies a coupling matrix (soft permutation) to a pytree.
        p_matrix has shape (B, B) and sums to 1 total.
        For barycentric projection, we want target[i] = sum_j (P_ij / mass_i) * x1[j]
        where mass_i = sum_j P_ij.
        In balanced OT with uniform weights 1/B, mass_i = 1/B.
        So target = B * P @ x1.
        """
        if pytree is None:
            return None
        batch_size = p_matrix.shape[0]
        # P matrix from OTT sums to 1 total. 
        # For barycentric projection: x1_coupled = B * P @ x1
        def _apply(x):
            x_flat = x.reshape(batch_size, -1)
            # Re-normalize rows of P to ensure it's a valid barycentric mapping 
            # even if Sinkhorn hasn't perfectly converged to marginals.
            row_sums = jnp.sum(p_matrix, axis=1, keepdims=True)
            p_normalized = p_matrix / jnp.where(row_sums > 0, row_sums, 1.0)
            return (p_normalized @ x_flat).reshape(x.shape)
            
        return jax.tree_util.tree_map(_apply, pytree)

class IndependentCoupling(Coupling):
    """
    p(x0, x1) = p(x0) p(x1)
    Randomly pairs samples within the batch.
    """
    def __call__(self, key, x0, x1, c0=None, c1=None, **extra_args):
        batch_size = jax.tree_util.tree_leaves(x0)[0].shape[0]
        perm = random.permutation(key, batch_size)
        
        x1_coupled = jax.tree_util.tree_map(lambda x: x[perm], x1)
        c1_coupled = jax.tree_util.tree_map(lambda x: x[perm], c1) if c1 is not None else None
        extra_args_coupled = jax.tree_util.tree_map(lambda x: x[perm], extra_args)
        
        return x0, x1_coupled, c0, c1_coupled, extra_args_coupled

class SinkhornCoupling(Coupling):
    """
    Computes a soft Optimal Transport coupling using the OTT-JAX Sinkhorn solver.
    Supports conditional OT if c0, c1 are provided.
    """
    # 'fixed': hand-rolled log-domain Sinkhorn with exactly n_iters iterations
    # (same g-then-f updates, zero init and uniform marginals as ott's balanced
    # lse_mode solver).  ott's solver checks convergence every 10 iterations
    # (threshold 1e-3), but at epsilon=0.1 on the benchmark batches it never
    # converges within 100 iterations (0/6000 batches), so it always runs all
    # 100; the fixed loop gives the same plan up to float32 rounding (~1e-4
    # relative) at ~4x lower cost.  'ott' keeps the original solver.
    BACKEND = 'fixed'

    def __init__(self, epsilon: float = 0.1, n_iters: int = 100, beta: float = 0.0, lse_mode: bool = True,
                 backend: str = None):
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.beta = beta
        self.lse_mode = lse_mode
        self.backend = backend

    @staticmethod
    def _fixed_sinkhorn_plan(x, y, epsilon, n_iters):
        n, m = x.shape[0], y.shape[0]
        C = jnp.sum(x ** 2, 1)[:, None] + jnp.sum(y ** 2, 1)[None, :] - 2 * x @ y.T
        la = jnp.log(jnp.full((n,), 1.0 / n))
        lb = jnp.log(jnp.full((m,), 1.0 / m))

        def body(_, fg):
            f, g = fg
            g = epsilon * lb - epsilon * jax.scipy.special.logsumexp((f[:, None] - C) / epsilon, axis=0)
            f = epsilon * la - epsilon * jax.scipy.special.logsumexp((g[None, :] - C) / epsilon, axis=1)
            return f, g
        f, g = jax.lax.fori_loop(0, n_iters, body, (jnp.zeros(n), jnp.zeros(m)), unroll=10)
        return jnp.exp((f[:, None] + g[None, :] - C) / epsilon)

    def __call__(self, key, x0, x1, c0=None, c1=None, **extra_args):
        x0_flat, _ = self._flatten_pytree(x0)
        x1_flat, _ = self._flatten_pytree(x1)
        
        v0 = x0_flat
        v1 = x1_flat
        
        if c0 is not None and c1 is not None and self.beta > 0:
            c0_flat, _ = self._flatten_pytree(c0)
            c1_flat, _ = self._flatten_pytree(c1)
            # Cost = ||x0-x1||^2 + beta * ||c0-c1||^2
            # We achieve this by concatenating sqrt(beta)*c to the vectors
            v0 = jnp.concatenate([v0, jnp.sqrt(self.beta) * c0_flat], axis=1)
            v1 = jnp.concatenate([v1, jnp.sqrt(self.beta) * c1_flat], axis=1)
            
        backend = self.backend or type(self).BACKEND
        if backend == 'fixed' and self.lse_mode:
            P = self._fixed_sinkhorn_plan(v0, v1, self.epsilon, self.n_iters)
        else:
            geom = pointcloud.PointCloud(v0, v1, epsilon=self.epsilon)
            prob = linear_problem.LinearProblem(geom)
            solver = sinkhorn.Sinkhorn(max_iterations=self.n_iters, lse_mode=self.lse_mode)
            out = solver(prob)
            P = out.matrix
        
        x1_coupled = self._apply_permutation(x1, P)
        c1_coupled = self._apply_permutation(c1, P) if c1 is not None else None
        extra_args_coupled = jax.tree_util.tree_map(lambda x: self._apply_permutation(x, P), extra_args)
        
        return x0, x1_coupled, c0, c1_coupled, extra_args_coupled

class ExactOTCoupling(SinkhornCoupling):
    """
    Standard Optimal Transport (Earth Mover's Distance).
    Approximated by Sinkhorn with small epsilon.
    """
    BACKEND = 'ott'  # small epsilon / 1000 iterations can converge early: keep ott's stopping rule

    def __init__(self, epsilon: float = 1e-2, n_iters: int = 1000, beta: float = 0.0, lse_mode: bool = True):
        super().__init__(epsilon=epsilon, n_iters=n_iters, beta=beta, lse_mode=lse_mode)

class ExactConditionalCoupling(ExactOTCoupling):
    """
    Conditional Wasserstein Coupling (defaults to high beta).
    """
    def __init__(self, beta: float = 100.0, epsilon: float = 1e-3, n_iters: int = 1000):
        super().__init__(beta=beta, epsilon=epsilon, n_iters=n_iters)


class C2OTCoupling(Coupling):
    """
    Condition-aware OT coupling (C2OT; Cheng & Schwing, ICCV 2025, arXiv:2503.10636).

    Minibatch OT pairs noise with data by distance alone, so the noise a conditional
    network is trained from ends up depending on the condition, while at sampling time
    it is drawn independently of it. C2OT adds a condition term to the cost,

        cost_ij = ||x0_i - x1_j||^2 + w * d(c0_i, c1_j),

    and solves it with an exact one-to-one (Hungarian) matching. Here x0 holds the
    data rows with conditions c0, and x1 the noise rows. Each noise row inherits the
    condition of the data row it was drawn alongside (c1 = c0 unless given), so the
    noise is only reordered between rows with similar conditions and stays close to
    independent of them. w = 0 is plain hard OT; w -> inf is the independent coupling
    whenever all conditions are distinct (continuous conditions).

    Args:
        w: fixed weight on the condition term. If None, w is chosen per batch so that
            the fraction of pairs (i, j) with cost_ij <= cost_ii is ``target_ratio``
            (the paper's adaptive rule, eq. 13; 0.5 ~ plain OT, 0 ~ independent).
        target_ratio: target for the adaptive rule (paper: 0.01).
        cond_dist: 'sqeuclidean' (default; suits low-dimensional numeric conditions),
            'cosine' (the paper's choice for embeddings; note it only sees the sign of
            a scalar condition), or 'discrete' (forbid any match between unequal
            conditions; w is ignored).
    """
    def __init__(self, w=None, target_ratio: float = 0.01, cond_dist: str = 'sqeuclidean'):
        if cond_dist not in ('sqeuclidean', 'cosine', 'discrete'):
            raise ValueError(f"unknown cond_dist {cond_dist!r}")
        self.w = w
        self.target_ratio = target_ratio
        self.cond_dist = cond_dist

    @staticmethod
    def _sqdist(a, b):
        return jnp.sum(a**2, 1)[:, None] + jnp.sum(b**2, 1)[None, :] - 2 * a @ b.T

    def _cond_cost(self, c0, c1):
        if self.cond_dist == 'sqeuclidean':
            return self._sqdist(c0, c1)
        if self.cond_dist == 'cosine':
            n0 = c0 / jnp.maximum(jnp.linalg.norm(c0, axis=1, keepdims=True), 1e-12)
            n1 = c1 / jnp.maximum(jnp.linalg.norm(c1, axis=1, keepdims=True), 1e-12)
            return 1.0 - n0 @ n1.T
        return jnp.any(c0[:, None, :] != c1[None, :, :], axis=-1).astype(c0.dtype)

    @staticmethod
    def _ratio(D, Cc, w):
        return np.mean(D + w * Cc <= np.diag(D)[:, None])

    @staticmethod
    def _adaptive_w_bisect(D, Cc, target):
        """Reference search for _adaptive_w (exponential bracket, then bisection)."""
        ratio = lambda w: C2OTCoupling._ratio(D, Cc, w)
        if ratio(0.0) <= target:
            return 0.0
        hi = max(float(np.mean(D)), 1e-12)
        for _ in range(60):                      # exponential search for an upper bound
            if ratio(hi) <= target:
                break
            hi *= 2.0
        lo = 0.0
        for _ in range(40):                      # bisection
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if ratio(mid) <= target else (mid, hi)
        return hi

    @staticmethod
    def _adaptive_w(D, Cc, target):
        """Smallest w with mean_ij 1(D_ij + w Cc_ij <= D_ii) <= target (numpy, host side).

        Closed form of the bisection above: a pair with Cc_ij > 0 counts iff w <= tau_ij =
        (D_ii - D_ij) / Cc_ij, so w sits just above the (k+1)-th largest tau, where k is the
        number of such pairs the target still allows."""
        D, Cc = np.asarray(D, np.float64), np.asarray(Cc, np.float64)
        diag = np.diag(D)[:, None]
        if np.mean(D <= diag) <= target:
            return 0.0
        pos = Cc > 0
        k = int(np.floor(target * D.size - np.sum(~pos & (D <= diag)) + 1e-9))
        tau = ((diag - D) / np.where(pos, Cc, 1.0))[pos]
        tau = tau[tau >= 0]
        if k < 0 or tau.size <= k:               # target unattainable: keep the reference behaviour
            return C2OTCoupling._adaptive_w_bisect(D, Cc, target)
        w = float(np.nextafter(np.partition(tau, tau.size - k - 1)[tau.size - k - 1], np.inf))
        for _ in range(8):                       # guard against rounding at the boundary
            if C2OTCoupling._ratio(D, Cc, w) <= target:
                return w
            w = float(np.nextafter(w, np.inf)) * (1 + 1e-12)
        return C2OTCoupling._adaptive_w_bisect(D, Cc, target)

    def _host_match(self, D, Cc, has_cond):
        """Hungarian matching for one cost pair (numpy). Returns (cols, w)."""
        from scipy.optimize import linear_sum_assignment
        D, Cc = np.asarray(D, np.float64), np.asarray(Cc, np.float64)
        if not has_cond:
            w, cost = 0.0, D
        elif self.cond_dist == 'discrete':
            w = 0.0
            cost = np.where(Cc > 0, 1e6 * (np.abs(D).max() + 1.0), D)
        else:
            w = (self._adaptive_w(D, Cc, self.target_ratio) if self.w is None else float(self.w))
            cost = D + w * Cc
        _, cols = linear_sum_assignment(cost)
        return cols.astype(np.int32), np.float32(w)

    def _costs(self, x0, x1, c0=None, c1=None):
        v0, _ = self._flatten_pytree(x0)
        v1, _ = self._flatten_pytree(x1)
        D = self._sqdist(v0, v1)
        if c0 is None:
            return D, jnp.zeros_like(D)
        f0, _ = self._flatten_pytree(c0)
        f1 = f0 if c1 is None else self._flatten_pytree(c1)[0]
        return D, self._cond_cost(f0, f1)

    def matching(self, x0, x1, c0=None, c1=None):
        """Returns (perm, w): data row i is paired with noise row perm[i]."""
        D, Cc = self._costs(x0, x1, c0, c1)
        out_shape = (jax.ShapeDtypeStruct((D.shape[0],), jnp.int32), jax.ShapeDtypeStruct((), jnp.float32))
        return jax.pure_callback(lambda D, Cc: self._host_match(D, Cc, c0 is not None), out_shape,
                                 jax.lax.stop_gradient(D), jax.lax.stop_gradient(Cc),
                                 vmap_method='sequential')

    def __call__(self, key, x0, x1, c0=None, c1=None, **extra_args):
        perm, _ = self.matching(x0, x1, c0, c1)
        take = lambda t: jax.tree_util.tree_map(lambda x: x[perm], t)
        c1_coupled = take(c1) if c1 is not None else None
        return x0, take(x1), c0, c1_coupled, take(extra_args)

    def couple_blocks(self, key, x_dict, e_dict, conditions):
        """
        Per-block C2OT for structured flows trained with block-local losses.

        For each block v, its noise column e_dict[v] is matched to x_dict[v] with the
        blocks in conditions[v] (e.g. its parents) as the condition, so that block v's
        noise stays independent of the variables its conditional field is given.
        Blocks with no conditions get plain hard OT. Each block uses its own matching, so
        a row's noise vector mixes rows of the original draw; it is still exactly N(0, I),
        because the blocks' noise columns are independent. Taken in a topological order,
        the population limit of this coupling is the Knothe-Rosenblatt map from the base to
        the data, which makes the joint flow-matching velocity itself block-triangular. That
        is why the dynamic (block-triangular) flow is consistent under this coupling and
        not under the joint or independent ones (Telfar & Sandfeld, TMLR 2026, appendix).
        All blocks are matched in one host call (one device-host round trip per step).
        """
        names = list(x_dict)
        has = [bool(conditions.get(n)) for n in names]
        costs = [self._costs(x_dict[n], e_dict[n], {p: x_dict[p] for p in conditions[n]} if h else None)
                 for n, h in zip(names, has)]
        D = jax.lax.stop_gradient(jnp.stack([c[0] for c in costs]))
        Cc = jax.lax.stop_gradient(jnp.stack([c[1] for c in costs]))

        def host(D, Cc):
            return np.stack([self._host_match(D[i], Cc[i], h)[0] for i, h in enumerate(has)])
        perms = jax.pure_callback(host, jax.ShapeDtypeStruct(D.shape[:2], jnp.int32), D, Cc,
                                  vmap_method='sequential')
        return {n: jax.tree_util.tree_map(lambda x: x[perms[i]], e_dict[n]) for i, n in enumerate(names)}
