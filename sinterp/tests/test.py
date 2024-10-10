import unittest

import jax.numpy as jnp
from jax import random
from sinterp.exact_si_gmm import construct_p_n_v

key = random.PRNGKey(0)

class TestExactSI(unittest.TestCase):
    def test_si(self):
        n = 200
        z = jnp.linspace(-5, 5, n)
        t = jnp.linspace(0, 1, n)

        d = 1
        K = 3
        m1s = [jnp.array([i]) for i in jnp.linspace(-K, K, K)]
        m2s = [jnp.array([i]) for i in jnp.linspace(-K, K, K)]

        c1s = [jnp.array([[0.05]]) for _ in range(K)]
        c2s = [jnp.array([[0.05]]) for _ in range(K)]

        w1s = jnp.ones(K) / K
        w2s = jnp.ones(K) / K

        p, v, s = construct_p_n_v(m1s, m2s, c1s, c2s, w1s, w2s)

        print(z[0:1].shape, t[0].shape)

        print(p(z[0:1], t[0]))
        print(v(z[0:1], t[0]))
        print(s(z[0:1], t[0]))

    def test_g_v_gmm(self):
        """
        check that the exact gaussian SI gives results equal to exact GMM with 1 mode. 
        """
        m1 = jnp.array([0.0])
        m2 = jnp.array([0.5])
        c1 = jnp.array([1.0])
        c2 = jnp.array([0.5])
        w1 = 1.0
        w2 = 1.0

        p0, v0, s0 = exact_si_gaussian.construt_p_v_s(m1, m2, c1, c2)
        p1, v1, s1 = exact_si_gmm.construt_p_v_s([m1], [m2], [c1], [c2], [w1], [w2])

        for _ in range(100):
            z = random.uniform(subkey, ()) * 10 - 5
            key, subkey = random.split(key)
            t = random.uniform(subkey, ())

            self.assertTrue(jnp.isclose(p0(z, t), p1(z, t), atol=1e-8))
            self.assertTrue(jnp.isclose(v0(z, t), v1(z, t), atol=1e-8))
            self.assertTrue(jnp.isclose(s0(z, t), s1(z, t), atol=1e-8))


class TestExactOT(unittest.TestCase):
    def test_ot(self):
        pass