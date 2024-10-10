import jax.numpy as jnp
from jax import vmap, jacrev, jit
from sinterp import utils
"""
page 35 stochastic interpolants. albergo et al 2022

exact calculation of the stoch interpolant between gaussian distributions.
supports any dimensionality, and any number of gaussian distributions.
"""

def interp_m(m1, m2, t):
    return m1 * (1 - t) + m2 * t

def interp_c(c1, c2, t, gamma=0.0):
    return c1 * (1 - t)**2 + c2 * t**2 + gamma*jnp.eye(c1.shape[0])

def construct_covar_interp(interp):
    def interp_c(c1, c2, t):
        return interp.alpha(t)**2 * c1 + interp.beta(t)**2 * c2 + interp.gamma(t)**2 * jnp.eye(c1.shape[0])
    return interp_c

def construct_p_b_s(interp, m1s, m2s, c1s, c2s, w1s, w2s):
    # assert m1.shape == m2.shape
    # assert c1.shape == c2.shape
    # assert c1.size == m1.size * m1.size

    e = jnp.zeros(m1s[0].shape)

    interp_c = construct_covar_interp(interp)

    dmtdt = jacrev(interp, argnums=3)
    dctdt = jacrev(interp_c, argnums=2)

    # vmap over gaussians in dist1. 
    i_gaussian_pdf = vmap(utils.gaussian_pdf, in_axes=(None, 0, 0))
    # vmap over gaussians in dist2.
    ij_gaussian_pdf = vmap(i_gaussian_pdf, in_axes=(None, 0, 0))
    
    n = len(m1s)
    m = len(m2s)
    d = m1s[0].size

    weight_ij = jnp.array([[w1s[i] * w2s[j] for i in range(n)] for j in range(m)])

    i_inv = vmap(jnp.linalg.inv, in_axes=(0,))
    ij_inv = vmap(i_inv, in_axes=(0,))

    def p(z, t):  # the density
        means = jnp.array([[interp(m1, m2, e, t) for m1 in m1s] for m2 in m2s]).reshape((m, n, 1))
        covs = jnp.array([[interp_c(c1, c2, t) for c1 in c1s] for c2 in c2s]).reshape((m, n, d, d))

        p_ij = ij_gaussian_pdf(z, means, covs)

        return jnp.sum(weight_ij * p_ij, axis=(0, 1))

    def s(z, t):  # the score
        """
        -C^{-1} (z - m)
        """
        means = jnp.array([[interp(m1, m2, e, t) for m1 in m1s] for m2 in m2s]).reshape((m, n, 1))
        covs = jnp.array([[interp_c(c1, c2, t) for c1 in c1s] for c2 in c2s]).reshape((m, n, d, d))

        c_m1 = ij_inv(covs)
        p_ij = ij_gaussian_pdf(z, means, covs)
        
        # vmap over the innermost function to ensure the matmul works
        inter = lambda c, diff: c @ diff
        i_inter = vmap(inter, in_axes=(0, 0))
        ij_inter = vmap(i_inter, in_axes=(0, 0))
        # for gaussians. score(z, t) = -C^-1 * (z - m)
        intermediate = -ij_inter(c_m1, z[None, None, :] - means)

        # now combine it all
        numer = jnp.sum(weight_ij[:, :, None] * intermediate
                      * p_ij[:, :, None], axis=(0, 1))
        denom = jnp.sum(weight_ij * p_ij, axis=(0, 1))

        return numer / denom

    def v(z, t):  # the velocity
        """
        \dot m - \frac{1}{2} \dot C C^{-1} (z - m)
        """
        means = jnp.array([[interp(m1, m2, e, t) for m1 in m1s] for m2 in m2s]).reshape((m, n, 1))
        covs = jnp.array([[interp_c(c1, c2, t) for c1 in c1s] for c2 in c2s]).reshape((m, n, d, d))

        dms_ij = jnp.array([[dmtdt(m1, m2, e, t) for m1 in m1s] for m2 in m2s]).reshape((m, n, 1))
        dcs_ij = jnp.array([[dctdt(c1, c2, t) for c1 in c1s] for c2 in c2s]).reshape((m, n, d, d))

        c_m1 = ij_inv(covs)
        p_ij = ij_gaussian_pdf(z, means, covs)
        
        # vmap over the innermost function to ensure the matmuls work
        inter = lambda dc, c, diff: 0.5 * dc @ c @ diff
        i_inter = vmap(inter, in_axes=(0, 0, 0))
        ij_inter = vmap(i_inter, in_axes=(0, 0, 0))
        intermediate = ij_inter(dcs_ij, c_m1, z[None, None, :] - means)

        # now combine it all
        numer = jnp.sum(weight_ij[:, :, None] * p_ij[:, :, None] * 
            (dms_ij + intermediate)  # velocity = m_dot + 0.5 * c_dot * c^-1 * (z - m)
                      , axis=(0, 1))
        denom = jnp.sum(weight_ij * p_ij, axis=(0, 1))

        return numer / denom
    
    
    return p, v, s