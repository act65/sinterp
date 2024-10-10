import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

def get_fokker_plank_eqns(b, s, eps=1.0):
    def forward(z, t):
        return b(z, t) + eps * s(z, t)
    def backward(z, t):
        return b(z, t) - eps * s(z, t)
    return forward, backward

def get_v(b, s, interp):
    def v(z, t):
        return b(z, t) + interp.dgammadt(t).flatten() * interp.gamma(t) * s(z, t)
    return v  

def extract_params(dist):
    return [d.mu for d in dist.dists], [d.std[None, None] for d in dist.dists], [d for d in dist.weights]

def gaussian_pdf(x, mean, cov):
    return jnp.exp(-0.5 * (x - mean) @ jnp.linalg.inv(cov) @ (x - mean)) * jnp.linalg.det(cov)**(-0.5) * (2 * jnp.pi)**(-0.5 * len(mean))

def wrap_plot(plt_fn, a, b, n, px, py):
    x = jnp.linspace(a, b, n)
    y = jnp.linspace(a, b, n)

    z = jnp.linspace(a, b, n)
    t = jnp.linspace(0, 1, n)

    plt.figure(figsize=(16, 8), dpi=300)
    f, (a0, a1, a2) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 5, 1]})

    plt.subplot(3, 1, 1)
    plt.plot(x, px.b_p(x[:, None]))
    plt.xlim([a, b])
    
    plt.subplot(3, 1, 3)
    plt.plot(y, py.b_p(y[:, None]))
    plt.xlim([a, b])

    plt.subplot(3, 1, 2)
    plt_fn(z, t, a, b)
    plt.xlim([a, b])

def plot_sf(z, t, a, b, p):
    p_z = p(z[:, None], t[:, None]).T
    plt.imshow(p_z, extent=[a, b, 1, 0], aspect='auto')
    plt.xlabel('z')
    plt.ylabel('t')

def plot_vf(z, t, a, b, k, v):

    # the vector field
    z = jnp.linspace(a, b, k)
    t = jnp.linspace(0, 1, k)

    Z, T = jnp.meshgrid(z, t)
    Z = Z.reshape(-1, 1)
    T = T.reshape(-1, 1)
    v_z = v(z[:, None], t[:, None])[:, :, 0].T

    plt.quiver(Z, T, v_z, -1*jnp.ones_like(v_z), width=0.001)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.ylabel('t')
    ax.axes.xaxis.set_visible(False)

def plot_trajectories(z, t, a, b, n, px, py, key, interp):
    key, subkey = random.split(key)
    xs = px.sample(subkey, n)
    key, subkey = random.split(key)
    ys = py.sample(subkey, n)

    k = t.shape[0]  # number of time steps
    e = random.normal(key, (n, k))
    lines = interp(xs, ys, e, t[None, :])

    for l in lines:
        plt.plot(l, t, linewidth=0.5, alpha=0.25)  # c='b')
        plt.xlim([a, b])

    ax = plt.gca()
    ax.invert_yaxis()
    ax.axes.xaxis.set_visible(False)
    plt.ylabel('t')