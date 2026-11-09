import jax
import jax.numpy as jnp
from jax import jacfwd

class Interpolator():
    def __init__(self):
        self.dalphadt = jacfwd(self.alpha, argnums=0)
        self.dbetadt = jacfwd(self.beta, argnums=0)
        self.dgammadt = jacfwd(self.gamma, argnums=0)

        self.didt = jacfwd(self.__call__, argnums=3)

    def alpha(self, t):
        raise NotImplementedError
    def beta(self, t):
        raise NotImplementedError
    def gamma(self, t):
        raise NotImplementedError
    
    def gamma_dgammadt(self, t):
        raise NotImplementedError
    
    def __call__(self, x, y, e, t):
        raise NotImplementedError
    
    def deterministic(self, x, y, t):
        e = jax.tree_util.tree_map(jnp.zeros_like, x)
        return self.__call__(x, y, e, t)
    def didt_deterministic(self, x, y, t):
        e = jax.tree_util.tree_map(jnp.zeros_like, x)
        return self.didt(x, y, e, t)
    
    def __repr__(self):
        return self.__class__.__name__

class LinearDeterministic(Interpolator):
    def __init__(self):
        super().__init__()

    def alpha(self, t):
        return (1-t)
    def beta(self, t):
        return t
    def gamma(self, t):
        return 0*t
    
    def __call__(self, x, y, z, t):
        return jax.tree_util.tree_map(lambda _x, _y, _z: self.alpha(t) * _x + self.beta(t) * _y + self.gamma(t) * _z, x, y, z)

class LinearStochastic(Interpolator):
    def __init__(self):
        super().__init__()

    def alpha(self, t):
        return jnp.where(t < 0.5, (1-2*t), 0)
    def beta(self, t):
        return jnp.where(t < 0.5, 0, 2*(t-0.5))
    def gamma(self, t):
        return jnp.where(t < 0.5, 2*t, 2*(1-t))
    
    def __call__(self, x, y, z, t):
        return jax.tree_util.tree_map(lambda _x, _y, _z: self.alpha(t) * _x + self.beta(t) * _y + self.gamma(t) * _z, x, y, z)

class SquaredDeterministic(Interpolator):
    def __init__(self):
        super().__init__()

    def alpha(self, t):
        return (1-t**2)
    def beta(self, t):
        return t**2
    def gamma(self, t):
        return 0*t
    
    def __call__(self, x, y, z, t):
        return jax.tree_util.tree_map(lambda _x, _y, _z: self.alpha(t) * _x + self.beta(t) * _y + self.gamma(t) * _z, x, y, z)

class SquaredStochastic(Interpolator):
    def __init__(self):
        super().__init__()

    def alpha(self, t):
        return jnp.where(t < 0.5, (1-2*t)**2, 0)
    
    def beta(self, t):
        return jnp.where(t < 0.5, 0, 4*(t-0.5)**2)
    
    def gamma(self, t):
        return jnp.where(t < 0.5, 4*t*(1-t), 4*(1-t)*t)
    
    def __call__(self, x, y, z, t):
        return jax.tree_util.tree_map(lambda _x, _y, _z: self.alpha(t) * _x + self.beta(t) * _y + self.gamma(t) * _z, x, y, z)
    
class TrigonometricStochastic(Interpolator):
    def __init__(self):
        super().__init__()

    def alpha(self, t):
        return jnp.sqrt(1 - self.gamma(t)**2) * jnp.cos(0.5 * jnp.pi * t)
    
    def beta(self, t):
        return jnp.sqrt(1 - self.gamma(t)**2) * jnp.sin(0.5 * jnp.pi * t)
    
    def gamma(self, t):
        return jnp.sqrt(2 * t * (1-t))
    
    def __call__(self, x, y, z, t):
        return jax.tree_util.tree_map(lambda _x, _y, _z: self.alpha(t) * _x + self.beta(t) * _y + self.gamma(t) * _z, x, y, z)
    
class LinearVPStochastic(Interpolator):
    """
    The 'Gold Standard' for Two-Sided Rectified Flows with Guidance.
    
    Path: Straight lines (alpha=1-t, beta=t).
    Noise: gamma(t) = sqrt(2t(1-t)).
    
    Properties:
    1. Straight mean paths (ideal for rectification/OT).
    2. Variance Preserving: (1-t)^2 + t^2 + 2t(1-t) = 1.
       This ensures consistent gradient magnitudes for guidance.
    3. Non-zero gamma: Allows for valid score estimation in TwoSidedField.
    """
    def __init__(self):
        super().__init__()

    def alpha(self, t):
        return 1.0 - t

    def beta(self, t):
        return t

    def gamma(self, t):
        # sqrt(2t(1-t))
        return jnp.sqrt(2 * t * (1 - t))

    def __call__(self, x, y, z, t):
        return jax.tree_util.tree_map(lambda _x, _y, _z: self.alpha(t) * _x + self.beta(t) * _y + self.gamma(t) * _z, x, y, z)


class TrigonometricVPStochastic(Interpolator):
    """
    A Variance-Preserving Trigonometric interpolant with a noise bridge.
    
    To strictly preserve variance (alpha^2 + beta^2 + gamma^2 = 1), 
    we must scale the cosine/sine terms by sqrt(1 - gamma^2).
    
    gamma(t) is parameterized as a simple sine bridge scaled by sigma.
    """
    def __init__(self, sigma=0.5):
        super().__init__()
        self.sigma = sigma

    def gamma(self, t):
        # A simple bridge that is 0 at t=0 and t=1
        return self.sigma * jnp.sin(jnp.pi * t)

    def alpha(self, t):
        # Standard VP schedule: cos(pi*t/2)
        # Scaled to make room for gamma variance
        scale = jnp.sqrt(1.0 - self.gamma(t)**2)
        return scale * jnp.cos(0.5 * jnp.pi * t)

    def beta(self, t):
        # Standard VP schedule: sin(pi*t/2)
        # Scaled to make room for gamma variance
        scale = jnp.sqrt(1.0 - self.gamma(t)**2)
        return scale * jnp.sin(0.5 * jnp.pi * t)

    def __call__(self, x, y, z, t):
        return jax.tree_util.tree_map(lambda _x, _y, _z: self.alpha(t) * _x + self.beta(t) * _y + self.gamma(t) * _z, x, y, z)

    
class EncodingDecodingStochastic(Interpolator):
    def __init__(self):
        super().__init__()

    def alpha(self, t):
        return jnp.where(t < 0.5, jnp.cos(jnp.pi * t)**2, 0)
    
    def beta(self, t):
        return jnp.where(t < 0.5, 0, jnp.cos(jnp.pi * t)**2)
    
    def gamma(self, t):
        return jnp.sin(jnp.pi * t)**2
    
    def __call__(self, x, y, z, t):
        return jax.tree_util.tree_map(lambda _x, _y, _z: self.alpha(t) * _x + self.beta(t) * _y + self.gamma(t) * _z, x, y, z)

class LowNoiseEncodingDecodingStochastic(Interpolator):
    def __init__(self):
        super().__init__()

    def alpha(self, t):
        return jnp.cos(0.5*jnp.pi * t)**4
    
    def beta(self, t):
        return jnp.sin(0.5*jnp.pi * t)**4
    
    def gamma(self, t):
        return 0.5*jnp.sin(jnp.pi * t)**2
    
    def __call__(self, x, y, z, t):
        return jax.tree_util.tree_map(lambda _x, _y, _z: self.alpha(t) * _x + self.beta(t) * _y + self.gamma(t) * _z, x, y, z)

class ConstantNoise(Interpolator):
    def __init__(self, sigma=1e-3):
        super().__init__()
        self.sigma = sigma

    def alpha(self, t):
        return (1-t)  # *(1-self.c)
    def beta(self, t):
        return t  #*(1-self.c)
    def gamma(self, t):
        return self.sigma + 0*t
    
    def __call__(self, x0, x1, e, t):
        return jax.tree_util.tree_map(lambda _x0, _x1, _e: self.alpha(t) * _x0 + self.beta(t) * _x1 + self.gamma(t) * _e, x0, x1, e)
    
class OneSidedLinear(Interpolator):
    """
    A simple one-sided interpolant: x_t = (1-t)e + t*x_1
    Bridges noise 'e' to data 'x_1'.
    """
    def alpha(self, t):
        return 0.0 * t # Always zero for this formulation

    def beta(self, t):
        return t

    def gamma(self, t):
        return 1.0 - t

    def __call__(self, x0, x1, e, t):
        # e is the start, x1 is the end
        # x0 is unused
        return jax.tree_util.tree_map(lambda _x1, _e: self.gamma(t) * _e + self.beta(t) * _x1, x1, e)

class OneSidedSBDM(Interpolator):
    """
    A variance-preserving one-sided interpolant used in SBDMs.
    x_t = sqrt(1-t^2)*z + t*x_1
    """
    def alpha(self, t):
        return 0.0 * t # Always zero

    def beta(self, t):
        # This is the data schedule
        return t

    def gamma(self, t):
        # This is the noise schedule, often written as sqrt(bar(alpha_t)) in papers
        return jnp.sqrt(1.0 - t**2)

    def __call__(self, x0, x1, e, t):
        return jax.tree_util.tree_map(lambda _x1, _e: self.gamma(t) * _e + self.beta(t) * _x1, x1, e)

interpolators = [
    LinearDeterministic,
    LinearStochastic,
    SquaredDeterministic,
    SquaredStochastic,
    EncodingDecodingStochastic,
    LowNoiseEncodingDecodingStochastic,
    ConstantNoise,
    TrigonometricStochastic,
    LinearVPStochastic,
    TrigonometricVPStochastic,
    OneSidedLinear,
    OneSidedSBDM
]
