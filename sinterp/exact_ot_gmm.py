import jax
import jax.numpy as jnp
from jax import vmap, jit
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

# -----------------------------------------------------------------------------
# 1. Geometry Helpers (Exact W2 Cost)
# -----------------------------------------------------------------------------

def sqrtm_psd(A):
    vals, vecs = jnp.linalg.eigh(A)
    vals = jnp.maximum(vals, 0.0)
    return vecs @ jnp.diag(jnp.sqrt(vals)) @ vecs.T

def compute_wasserstein_cost_matrix(m0s, m1s, c0s, c1s):
    """
    Computes the exact W2^2 cost matrix between Gaussian components.
    C_ij = ||m0_i - m1_j||^2 + Tr(C0_i + C1_j - 2(C0_i^0.5 C1_j C0_i^0.5)^0.5)
    """
    # Euclidean distance between means
    diff_means = m0s[:, None, :] - m1s[None, :, :]
    dist_means = jnp.sum(diff_means**2, axis=-1)
    
    # Bures metric for covariances
    def bures_trace(c0, c1):
        c0_sqrt = sqrtm_psd(c0)
        inner = c0_sqrt @ c1 @ c0_sqrt
        return jnp.trace(c0 + c1 - 2 * sqrtm_psd(inner))

    vmap_bures = vmap(vmap(bures_trace, in_axes=(None, 0)), in_axes=(0, None))
    dist_covs = vmap_bures(c0s, c1s)
    
    return dist_means + dist_covs

# -----------------------------------------------------------------------------
# 2. OTT Integration
# -----------------------------------------------------------------------------
@jit
def get_ott_coupling(m0s, m1s, c0s, c1s, w0s, w1s, epsilon=0.1):
    """
    Solves the Optimal Transport coupling matrix using OTT-JAX.
    """
    # 1. Compute the Custom Cost Matrix (Exact W2)
    #    We cannot use PointCloud because it ignores covariance.
    cost_matrix = compute_wasserstein_cost_matrix(m0s, m1s, c0s, c1s)
    
    # 2. Define Geometry using this cost
    geom = geometry.Geometry(cost_matrix=cost_matrix, epsilon=epsilon)
    
    # 3. Define Problem with weights (marginals)
    #    Ensure weights sum to 1
    a = w0s / jnp.sum(w0s)
    b = w1s / jnp.sum(w1s)
    prob = linear_problem.LinearProblem(geom, a=a, b=b)
    
    # 4. Solve
    solver = sinkhorn.Sinkhorn()
    out = solver(prob)
    
    # Return the coupling matrix P_ij
    return out.matrix

# -----------------------------------------------------------------------------
# 3. Dynamic Interpolant Construction
# -----------------------------------------------------------------------------

def log_gaussian_pdf(x, mean, cov):
    d = x.shape[-1]
    diff = x - mean
    # Stable solve
    L = jnp.linalg.cholesky(cov + jnp.eye(d)*1e-6)
    sol = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
    mahal = jnp.sum(sol**2)
    const = d * jnp.log(2 * jnp.pi) + 2 * jnp.sum(jnp.log(jnp.diag(L)))
    return -0.5 * (const + mahal)

def construct_optimal_transport(m0s, m1s, c0s, c1s, w0s, w1s, epsilon=0.1):
    
    # --- Step 1: Get the Coupling Matrix via OTT ---
    pi_ij = get_ott_coupling(m0s, m1s, c0s, c1s, w0s, w1s, epsilon)
    
    # Dimensions
    N, d = m0s.shape
    M = m1s.shape[0]
    I = jnp.eye(d)

    # --- Step 2: Precompute Geodesic Maps (McCann) ---
    # T(x) = m1 + A(x - m0) where A minimizes transport cost between covariances
    def get_map_A(c0, c1):
        c0_sqrt = sqrtm_psd(c0)
        c0_inv_sqrt = jnp.linalg.inv(c0_sqrt)
        middle = sqrtm_psd(c0_sqrt @ c1 @ c0_sqrt)
        return c0_inv_sqrt @ middle @ c0_inv_sqrt

    As = vmap(vmap(get_map_A, in_axes=(None, 0)), in_axes=(0, None))(c0s, c1s)

    # --- Step 3: Define Dynamic Functions ---
    
    def get_pair_stats(z, t, i, j):
        m0, m1 = m0s[i], m1s[j]
        c0 = c0s[i]
        A = As[i, j]
        
        # Geodesic Interpolation
        mu_t = (1 - t) * m0 + t * m1
        M_t = (1 - t) * I + t * A # Transport Map Jacobian
        cov_t = M_t @ c0 @ M_t.T
        
        # Density
        log_pdf = log_gaussian_pdf(z, mu_t, cov_t)
        p_val = jnp.exp(log_pdf)
        
        # Velocity v_t(x) = (m1 - m0) + (A - I) M_t^{-1} (x - mu_t)
        diff_z = z - mu_t
        y = jnp.linalg.solve(M_t, diff_z)
        vel_val = (m1 - m0) + (A - I) @ y
        
        # Score s_t(x) = -C_t^{-1} (x - mu_t)
        c0_inv_y = jnp.linalg.solve(c0, y)
        score_val = -jnp.linalg.solve(M_t.T, c0_inv_y)
        
        return p_val, vel_val, score_val

    # Vmap over all pairs
    vmap_stats = vmap(vmap(get_pair_stats, in_axes=(None, None, None, 0)), 
                      in_axes=(None, None, 0, None))

    def compute_mixture(z, t):
        # 1. Compute p, v, s for every pair (i,j)
        p_vals, v_vals, s_vals = vmap_stats(z, t, jnp.arange(N), jnp.arange(M))
        
        # 2. Weight by the OT Coupling Matrix
        weighted_p = pi_ij * p_vals
        denom = jnp.sum(weighted_p)
        
        # 3. Aggregate
        safe_denom = jnp.where(denom < 1e-60, 1.0, denom)
        b_z = jnp.sum(weighted_p[:, :, None] * v_vals, axis=(0, 1)) / safe_denom
        s_z = jnp.sum(weighted_p[:, :, None] * s_vals, axis=(0, 1)) / safe_denom
        
        return denom, b_z, s_z

    def p(z, t): return compute_mixture(z, t)[0]
    def b(z, t): return compute_mixture(z, t)[1]
    def s(z, t): return compute_mixture(z, t)[2]

    return p, b, s