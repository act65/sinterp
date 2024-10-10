import jax.numpy as jnp
from sinterp.interpolants import Interpolator

def chex_wrapper(fn):
    # TODO use chex...
    def wrapped(params, x, y, e, t):
        assert len(t.shape) == 0
        assert x.shape == y.shape == e.shape
        return fn(params, x, y, e, t)
    return wrapped

def make_loss_b(interp: Interpolator, b):
    @chex_wrapper
    def loss_b(params, x, y, e, t):
        assert len(t.shape) == 0
        i = interp(x, y, e, t)
        di = interp.didt(x, y, e, t)

        v = b(params, i, t)

        if v.shape != di.shape:
            raise ValueError(f"v.shape: {v.shape}, di.shape: {di.shape}")
        
        return jnp.mean((v - di)**2)
    
    return loss_b

def make_loss_v(interp: Interpolator, vel):
    @chex_wrapper
    def loss_v(params, x, y, e, t):
        i = interp(x, y, e, t)  # this should also be deterministic?
        di = interp.didt_deterministic(x, y, t)

        v = vel(params, i, t)

        if v.shape != di.shape:
            raise ValueError(f"v.shape: {v.shape}, di.shape: {di.shape}")
        return jnp.mean((v - di)**2)
    
    return loss_v

def make_loss_s(interp: Interpolator, score_fn: callable):
    @chex_wrapper
    def loss_s(params, x, y, e, t):
        # https://github.com/malbergo/stochastic-interpolants/blob/main/interflow/stochastic_interpolant.py#L701
		# loss = 0.5*torch.sum(st**2) + (1 / interpolant.gamma(t))*torch.sum(st*x0)
        assert len(t.shape) == 0
        i = interp.deterministic(x, y, t)
        di = interp.didt_deterministic(x, y, t)

        st = score_fn(params, i, t)

        return 0.5 * jnp.linalg.norm(st)**2 + (1/interp.gamma(t)) * ((st * x).sum())
    
    return loss_s