import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List

class FourierFeatures(eqx.Module):
    """Encodes time with Fourier features."""
    w: jax.Array
    output_features: int = eqx.field(static=True)
    trainable: bool = eqx.field(static=True)

    def __init__(self, output_features: int, key: jax.Array, trainable: bool = True):
        self.output_features = output_features
        self.trainable = trainable
        
        half_dim = output_features // 2
        
        if self.trainable:
            # Learnable Fourier features (stddev=1.0)
            self.w = jax.random.normal(key, (half_dim, 1))
        else:
            # Fixed Fourier features
            self.w = jnp.exp(
                jnp.linspace(jnp.log(1.0), jnp.log(1000.0), half_dim)
            )[:, None]

    def __call__(self, t):
        # In Equinox, if it's not trainable, we usually rely on the optimizer 
        # filtering based on equinox.partition, but to be safe and match logic 
        # strictly, we can use stop_gradient if trainable=False.
        w = self.w
        if not self.trainable:
            w = jax.lax.stop_gradient(w)
            
        t = jnp.atleast_1d(t)
        t_proj = t * w
        return jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=0)


class VelocityNet(eqx.Module):
    """A simple MLP to model a velocity b(x, t) with near-zero initialization."""
    time_embed: FourierFeatures
    input_layer: eqx.nn.Linear
    hidden_layers: List[eqx.nn.Linear]
    output_layer: eqx.nn.Linear
    
    dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    time_embedding_dim: int = eqx.field(static=True)

    def __init__(
        self, 
        dim: int, 
        key: jax.Array, 
        depth: int = 4, 
        width: int = 256, 
        time_embedding_dim: int = 32
    ):
        self.dim = dim
        self.depth = depth
        self.width = width
        self.time_embedding_dim = time_embedding_dim

        # Split keys for each layer
        keys = jax.random.split(key, depth + 3)

        # 1. Time Embedding
        self.time_embed = FourierFeatures(
            output_features=time_embedding_dim, 
            key=keys[0]
        )

        # 2. Input Layer (concatenated x + t_emb)
        self.input_layer = eqx.nn.Linear(dim + time_embedding_dim, width, key=keys[1])

        # 3. Hidden Layers
        self.hidden_layers = [
            eqx.nn.Linear(width, width, key=k) 
            for k in keys[2:2+depth]
        ]

        # 4. Output Layer with Zero Init
        # Equinox layers default to LeCun/Uniform init. We manually set to zero.
        out_layer = eqx.nn.Linear(width, dim, key=keys[-1])
        zeros_w = jnp.zeros((dim, width))
        zeros_b = jnp.zeros((dim,))
        
        # Replace weights and biases with zeros
        self.output_layer = eqx.tree_at(
            lambda l: (l.weight, l.bias), 
            out_layer, 
            (zeros_w, zeros_b)
        )

    def __call__(self, x, t):
        t = jnp.atleast_1d(t)
        assert x.ndim == 1
        assert t.ndim == 1

        t_emb = self.time_embed(t).squeeze()
        inputs = jnp.concatenate([x, t_emb], axis=-1)

        h = self.input_layer(inputs)
        
        for layer in self.hidden_layers:
            h = jax.nn.relu(h)
            h = layer(h)

        out = self.output_layer(h)
        return out

class DualOutputNet(eqx.Module):
    """
    A single backbone with two heads:
    1. Velocity Head (v)
    2. Denoiser Head (eta_z)
    """
    time_embed: FourierFeatures
    input_layer: eqx.nn.Linear
    hidden_layers: List[eqx.nn.Linear]
    head_v: eqx.nn.Linear
    head_etaz: eqx.nn.Linear
    
    dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    time_embedding_dim: int = eqx.field(static=True)

    def __init__(
        self, 
        dim: int, 
        key: jax.Array, 
        depth: int = 4, 
        width: int = 256, 
        time_embedding_dim: int = 32
    ):
        self.dim = dim
        self.depth = depth
        self.width = width
        self.time_embedding_dim = time_embedding_dim

        keys = jax.random.split(key, depth + 4) # +4 for time, input, head1, head2

        # Shared Backbone Setup
        self.time_embed = FourierFeatures(output_features=time_embedding_dim, key=keys[0])
        self.input_layer = eqx.nn.Linear(dim + time_embedding_dim, width, key=keys[1])
        
        self.hidden_layers = [
            eqx.nn.Linear(width, width, key=k) 
            for k in keys[2:2+depth]
        ]

        # Helper to create zero-initialized layer
        def make_zero_layer(in_f, out_f, key):
            l = eqx.nn.Linear(in_f, out_f, key=key)
            return eqx.tree_at(
                lambda x: (x.weight, x.bias), 
                l, 
                (jnp.zeros((out_f, in_f)), jnp.zeros((out_f,)))
            )

        # --- Head 1: Velocity ---
        self.head_v = make_zero_layer(width, dim, keys[-2])

        # --- Head 2: Denoiser ---
        self.head_etaz = make_zero_layer(width, dim, keys[-1])

    def __call__(self, x, t):
        t = jnp.atleast_1d(t)
        
        # Shared Time Embedding
        t_emb = self.time_embed(t).squeeze()
        
        # Concatenate
        inputs = jnp.concatenate([x, t_emb], axis=-1)

        # Shared Backbone
        h = self.input_layer(inputs)
        for layer in self.hidden_layers:
            h = jax.nn.relu(h)
            h = layer(h)

        # Output Heads
        v_out = self.head_v(h)
        eta_z_out = self.head_etaz(h)

        return v_out, eta_z_out
    
class EtaNet(eqx.Module):
    """A simple MLP to model a eta_1(x, t)"""
    time_embed: FourierFeatures
    input_layer: eqx.nn.Linear
    hidden_layers: List[eqx.nn.Linear]
    output_layer: eqx.nn.Linear
    
    dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    time_embedding_dim: int = eqx.field(static=True)

    def __init__(
        self, 
        dim: int, 
        key: jax.Array, 
        depth: int = 4, 
        width: int = 256, 
        time_embedding_dim: int = 32
    ):
        self.dim = dim
        self.depth = depth
        self.width = width
        self.time_embedding_dim = time_embedding_dim

        # Split keys for each layer
        keys = jax.random.split(key, depth + 3)

        # 1. Time Embedding
        self.time_embed = FourierFeatures(
            output_features=time_embedding_dim, 
            key=keys[0]
        )

        # 2. Input Layer (concatenated x + t_emb)
        self.input_layer = eqx.nn.Linear(dim + time_embedding_dim, width, key=keys[1])

        # 3. Hidden Layers
        self.hidden_layers = [
            eqx.nn.Linear(width, width, key=k) 
            for k in keys[2:2+depth]
        ]

        # 4. Output Layer
        self.output_layer = eqx.nn.Linear(width, dim, key=keys[-1])

    def __call__(self, x, t):
        t = jnp.atleast_1d(t)
        assert x.ndim == 1
        assert t.ndim == 1

        t_emb = self.time_embed(t).squeeze()
        inputs = jnp.concatenate([x, t_emb], axis=-1)

        h = self.input_layer(inputs)
        
        for layer in self.hidden_layers:
            h_hat = jax.nn.relu(h)
            h += layer(h_hat)  # residual connections

        out = self.output_layer(h)
        return out+x  # unet