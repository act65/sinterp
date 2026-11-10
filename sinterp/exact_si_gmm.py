import jax
import jax.numpy as jnp
from jax import vmap, jacrev, jit
from sinterp import utils
"""
page 35 stochastic interpolants. albergo et al 2022

exact calculation of the stoch interpolant between gaussian distributions.
supports any dimensionality, and any number of gaussian distributions.
"""

def construct_covar_interp(interp):
    def interp_c(c0, c1, ce, t):
        return interp.alpha(t)**2 * c0 + interp.beta(t)**2 * c1 + interp.gamma(t)**2 * ce
    return interp_c

def construct_p_b_s(interp, m0s, m1s, c0s, c1s, w0s, w1s, ce=None):
    # assert m0.shape == m1.shape
    # assert c0.shape == c1.shape
    # assert c0.size == m0.size * m0.size

    if ce is None:
        ce = jnp.eye(m0s[0].shape[0])

    e = jnp.zeros(m0s[0].shape)

    interp_c = construct_covar_interp(interp)

    dmtdt = jacrev(interp, argnums=3)
    dctdt = jacrev(interp_c, argnums=3)

    # vmap over gaussians in dist1. 
    i_gaussian_pdf = vmap(utils.gaussian_pdf, in_axes=(None, 0, 0))
    i_gaussian_logpdf = vmap(utils.gaussian_logpdf, in_axes=(None, 0, 0))
    # vmap over gaussians in dist2.
    ij_gaussian_pdf = vmap(i_gaussian_pdf, in_axes=(None, 0, 0))
    ij_gaussian_logpdf = vmap(i_gaussian_logpdf, in_axes=(None, 0, 0))
    
    n = len(m0s)
    m = len(m1s)
    d = m0s[0].size

    weight_ij = jnp.array([[w0s[i] * w1s[j] for i in range(n)] for j in range(m)])

    def responsibilities(z, means, covs):
        """Posterior weights of the (j, i) components at z, a softmax of log-densities, so
        they stay finite far from every component (ratios of raw densities underflow to 0/0)."""
        logits = jnp.log(weight_ij) + ij_gaussian_logpdf(z, means, covs)
        return jax.nn.softmax(logits.ravel()).reshape(logits.shape)

    i_inv = vmap(jnp.linalg.inv, in_axes=(0,))
    ij_inv = vmap(i_inv, in_axes=(0,))

    def p(z, t):  # the density
        means = jnp.array([[interp(m0, m1, e, t) for m0 in m0s] for m1 in m1s]).reshape((m, n, d))
        covs = jnp.array([[interp_c(c0, c1, ce, t) for c0 in c0s] for c1 in c1s]).reshape((m, n, d, d))

        p_ij = ij_gaussian_pdf(z, means, covs)

        return jnp.sum(weight_ij * p_ij, axis=(0, 1))

    def s(z, t):  # the score
        """
        -C^{-1} (z - m)
        """
        means = jnp.array([[interp(m0, m1, e, t) for m0 in m0s] for m1 in m1s]).reshape((m, n, d))
        covs = jnp.array([[interp_c(c0, c1, ce, t) for c0 in c0s] for c1 in c1s]).reshape((m, n, d, d))

        c_m1 = ij_inv(covs)
        r_ij = responsibilities(z, means, covs)
        
        # vmap over the innermost function to ensure the matmul works
        inter = lambda c, diff: c @ diff
        i_inter = vmap(inter, in_axes=(0, 0))
        ij_inter = vmap(i_inter, in_axes=(0, 0))
        # for gaussians. score(z, t) = -C^-1 * (z - m)
        intermediate = -ij_inter(c_m1, z[None, None, :] - means)

        # now combine it all
        return jnp.sum(r_ij[:, :, None] * intermediate, axis=(0, 1))

    def b(z, t):  # the velocity
        r"""
        \dot m - \frac{1}{2} \dot C C^{-1} (z - m)
        """
        means = jnp.array([[interp(m0, m1, e, t) for m0 in m0s] for m1 in m1s]).reshape((m, n, d))
        covs = jnp.array([[interp_c(c0, c1, ce, t) for c0 in c0s] for c1 in c1s]).reshape((m, n, d, d))

        dms_ij = jnp.array([[dmtdt(m0, m1, e, t) for m0 in m0s] for m1 in m1s]).reshape((m, n, d))
        dcs_ij = jnp.array([[dctdt(c0, c1, ce, t) for c0 in c0s] for c1 in c1s]).reshape((m, n, d, d))

        c_m1 = ij_inv(covs)
        r_ij = responsibilities(z, means, covs)
        
        # vmap over the innermost function to ensure the matmuls work
        inter = lambda dc, c, diff: 0.5 * dc @ c @ diff
        i_inter = vmap(inter, in_axes=(0, 0, 0))
        ij_inter = vmap(i_inter, in_axes=(0, 0, 0))
        intermediate = ij_inter(dcs_ij, c_m1, z[None, None, :] - means)

        # now combine it all
        # velocity = m_dot + 0.5 * c_dot * c^-1 * (z - m), averaged over the responsibilities
        return jnp.sum(r_ij[:, :, None] * (dms_ij + intermediate), axis=(0, 1))
    
    
    return p, b, s