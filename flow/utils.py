import jax
import jax.numpy as jnp
from jax import vmap, lax
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree

class FlattenFn:
    def __init__(self, fn, argnum=0):
        self.fn = fn
        self.unflatten = None
        self.argnum = argnum

    def __call__(self, *args):
        # args[self.argnum] is now flat, we need to unflatten it
        unflattened_arg = self.unflatten(args[self.argnum])
        new_args = list(args)
        new_args[self.argnum] = unflattened_arg
        
        # Call the original function
        out = self.fn(*new_args)
        
        # Ravel the output
        flat_out, _ = ravel_pytree(out)
        return flat_out

    def flatten_in(self, *args):
        flat_arg, unflatten = ravel_pytree(args[self.argnum])
        self.unflatten = unflatten
        new_args = list(args)
        new_args[self.argnum] = flat_arg
        return new_args

def divergence_jvp(fn, argnums=0):
    """
    Exact divergence using JVP and vmap (slower than Hutchinson, exact result).
    """
    def _divergence(*args, **kwargs):
        flat_fn = FlattenFn(fn, argnum=argnums)
        flat_args = flat_fn.flatten_in(*args)
        primals_flat = flat_args[argnums]
        n = len(primals_flat)

        def get_diag_element(i):
            basis_vector = jax.nn.one_hot(i, n, dtype=primals_flat.dtype)
            
            # Create zero tangents for all args
            tangent_args = [jax.tree_util.tree_map(jnp.zeros_like, p) for p in flat_args]
            # Inject basis vector into the argument we are differentiating
            tangent_args[argnums] = basis_vector
            
            _, tangent_out = jax.jvp(flat_fn, flat_args, tangent_args)
            return tangent_out[i]

        return jnp.sum(vmap(get_diag_element)(jnp.arange(n)))
    return _divergence

def divergence_hutchinson(fn, argnums=0):
    """
    Stochastic divergence estimator.
    Requires 'key' and 'step_idx' to be passed as kwargs to the returned function.
    """
    def _divergence(*args, key, step_idx=0, **kwargs):
        flat_fn = FlattenFn(fn, argnum=argnums)
        flat_args = flat_fn.flatten_in(*args)
        primals_flat = flat_args[argnums]
        
        # 1. Manage Randomness: Fold step_idx into key for consistent noise per step
        rng = jax.random.fold_in(key, step_idx)
        
        # 2. Sample noise
        epsilon = jax.random.normal(rng, shape=primals_flat.shape)
        
        # 3. Construct tangents
        tangent_args = [jax.tree_util.tree_map(jnp.zeros_like, p) for p in flat_args]
        tangent_args[argnums] = epsilon
        
        # 4. Compute JVP
        _, tangent_out = jax.jvp(flat_fn, flat_args, tangent_args)
        
        # 5. Trace estimator: e^T * (J * e)
        return jnp.dot(epsilon, tangent_out)
    return _divergence

def scan_with_init_state(fn, init, n, length):
    """
    jnp.lax.scan doesnt return the initial state.
    This function does.
    """
    _,  xs = lax.scan(fn, init, n[:-1], length=length-1)
    return tree_map(lambda i, x: jnp.concatenate([i[None, ...], x], axis=0), init, xs)

def zip_arrays(a, b):
    """
    Given two arrays of the same shape (n, ...).
    Return a single array of shape (2 * n, ...).
    Where the values are zipped together yielding [a_0, b_0, a_1, b_1, ..., a_n, b_n]
    """
    assert a.shape == b.shape
    return jnp.concatenate([a[None, ...], b[None, ...]], axis=0).reshape(-1, *a.shape[1:], order='F')

def scan_wrapper(fn):
    def _fn(acc, x):
        y = fn(x, acc)
        return y, y
    return _fn
