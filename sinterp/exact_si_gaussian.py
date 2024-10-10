import jax.numpy as jnp
from jax import jacrev
from sinterp.utils import gaussian_pdf
from sinterp.interpolants import LinearDeterministic

def construct_covar_interp(interp):
    def interp_c(c1, c2, t):
        return interp.alpha(t)**2 * c1 + interp.beta(t)**2 * c2 + interp.gamma(t)**2 * jnp.eye(c1.shape[0])
    return interp_c

def construct_p_b_s(m1, m2, c1, c2, interp):
    assert m1.shape == m2.shape
    assert c1.shape == c2.shape
    assert c1.size == m1.size * m1.size

    interp_c = construct_covar_interp(interp)

    dmtdt = jacrev(interp, argnums=3)
    dctdt = jacrev(interp_c, argnums=2)

    def p(z, t):  # the density
        
        mt = interp(m1, m2, 0.0, t)
        ct = interp_c(c1, c2, t) 

        return gaussian_pdf(z, mt, ct)

    def v(z, t):  # the velocity

        mt = interp(m1, m2, 0.0, t).reshape((m1.size, ))
        ct = interp_c(c1, c2, t).reshape((m1.size, m1.size))

        dm = dmtdt(m1, m2, 0.0, t).reshape((m1.size, ))
        dc = dctdt(c1, c2, t).reshape((m1.size, m1.size))

        return dm + 0.5 * dc @ jnp.linalg.pinv(ct) @ (z - mt)
    
    def s(z, t):  # the score

        mt = interp(m1, m2, 0.0, t).reshape((m1.size, ))
        ct = interp_c(c1, c2, t).reshape((m1.size, m1.size))

        return - jnp.linalg.solve(ct, z - mt).reshape((m1.size, ))

    return p, v, s