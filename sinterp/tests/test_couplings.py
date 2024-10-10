import unittest

from sinterp.couplings import IndependentCoupling, EMDCoupling, ConditionalCoupling

import jax.numpy as jnp 
from jax import random, grad, jit, vmap

# import ot as pot

import matplotlib.pyplot as plt

class TestCouplings(unittest.TestCase):
    def test_emd(self):
        key = random.PRNGKey(1)
        n = 500

        key, subkey = random.split(key)
        x = random.normal(subkey, (n, 1))
        key, subkey = random.split(key)
        y = random.normal(subkey, (n, 1))

        coupling = EMDCoupling()

        x_hat, y_hat = coupling(x, y, key)

        plt.scatter(x, y)
        plt.scatter(x_hat, y_hat)
        plt.show()

    def test_grad(self):
        """
        getting jax.errors.TracerArrayConversionError
        """

        key = random.PRNGKey(0)
        B = 10

        key, subkey = random.split(key)
        x = random.normal(subkey, (B, 1))
        key, subkey = random.split(key)
        y = random.normal(subkey, (B, 1))


        def loss_fn(x, y):
            costs = jnp.linalg.norm(x[:, None] - y[None, :], axis=-1)**2

            # pi = pot.emd(
            #     jnp.ones(B) / B, 
            #     jnp.ones(B) / B, 
            #     costs)
            
            l = pot.emd2(
                jnp.ones(B) / B, 
                jnp.ones(B) / B, 
                costs)

            return l

        g = vmap(loss_fn)(x, y)
        print(g)