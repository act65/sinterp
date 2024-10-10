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
        e = jnp.zeros_like(x)
        return self.__call__(x, y, e, t)
    def didt_deterministic(self, x, y, t):
        e = jnp.zeros_like(x)
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
        return self.alpha(t) * x + self.beta(t) * y + self.gamma(t) * z

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
        return self.alpha(t) * x + self.beta(t) * y + self.gamma(t) * z

class SquaredDeterministic(Interpolator):
    def __init__(self):
        self.dalphadt = jacfwd(self.alpha, argnums=0)
        self.dbetadt = jacfwd(self.beta, argnums=0)
        self.dgammadt = jacfwd(self.gamma, argnums=0)

        self.didt = jacfwd(self.__call__, argnums=3)

    def alpha(self, t):
        return (1-t**2)
    def beta(self, t):
        return t**2
    def gamma(self, t):
        return 0*t
    
    def __call__(self, x, y, z, t):
        return self.alpha(t) * x + self.beta(t) * y + self.gamma(t) * z

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
        return self.alpha(t) * x + self.beta(t) * y + self.gamma(t) * z
    
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
        return self.alpha(t) * x + self.beta(t) * y + self.gamma(t) * z
    
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
        return self.alpha(t) * x + self.beta(t) * y + self.gamma(t) * z

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
        return self.alpha(t) * x + self.beta(t) * y + self.gamma(t) * z

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
    
    def __call__(self, x, y, z, t):
        return self.alpha(t) * x + self.beta(t) * y + self.gamma(t) * z

interpolators = [
    LinearDeterministic,
    LinearStochastic,
    SquaredDeterministic,
    SquaredStochastic,
    EncodingDecodingStochastic,
    LowNoiseEncodingDecodingStochastic,
    ConstantNoise,
]

def get_interp(name, sigma=1e-3):
    if name == "LinearStochastic":
        return LinearStochastic()
    elif name == "LinearDeterministic":
        return LinearDeterministic()
    elif name == "SquaredStochastic":
        return SquaredStochastic()
    elif name == "SquaredDeterministic":
        return SquaredDeterministic()
    elif name == "EncodingDecodingStochastic":
        return EncodingDecodingStochastic()
    elif name == "lower-noise":
        return LowNoiseEncodingDecodingStochastic()
    elif name == "ConstantNoise":
        return ConstantNoise(sigma=sigma)
    else:
        raise ValueError(f"Unknown interp name: {name}")