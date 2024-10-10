import jax.numpy as jnp
from jax import random, vmap, jit
from sinterp.exact_si_gmm import construct_p_b_s
from sinterp.interpolants import get_interp
from sinterp.utils import wrap_plot, plot_sf, plot_vf, plot_trajectories, extract_params, get_v

from exp_utils.data.gaussians import many_g

import matplotlib.pyplot as plt
import os
from functools import partial

import fire

def main(savedir, interp_name):
    m = 3
    n = 2

    px, py = many_g(m, n, std=0.1)
    mxs, cxs, wxs = extract_params(px)
    mys, cys, wys = extract_params(py)

    A = -5
    B = 5

    N = 200

    interp = get_interp(interp_name)

    k = 30

    p, b, s = construct_p_b_s(interp, mxs, mys, cxs, cys, wxs, wys)
    v = get_v(b, s, interp)

    p = jit(vmap(vmap(p, in_axes=(None, 0)), in_axes=(0, None)))
    b = jit(vmap(vmap(b, in_axes=(None, 0)), in_axes=(0, None)))
    s = jit(vmap(vmap(s, in_axes=(None, 0)), in_axes=(0, None)))
    v = jit(vmap(vmap(v, in_axes=(None, 0)), in_axes=(0, None)))

    wrap_plot(partial(plot_sf, p=p), A, B, N, px, py)
    plt.savefig(os.path.join(savedir, f'p_{interp_name}-{m}-{n}.png'))
    wrap_plot(partial(plot_vf, k=k, v=b), A, B, N, px, py)
    plt.savefig(os.path.join(savedir, f'b_{interp_name}-{m}-{n}.png'))
    wrap_plot(partial(plot_vf, k=k, v=s), A, B, N, px, py)
    plt.savefig(os.path.join(savedir, f's_{interp_name}-{m}-{n}.png'))
    wrap_plot(partial(plot_vf, k=k, v=v), A, B, N, px, py)
    plt.savefig(os.path.join(savedir, f'v_{interp_name}-{m}-{n}.png'))

    
    key = random.PRNGKey(0)
    wrap_plot(partial(plot_trajectories, n=k*10, px=px, py=py, key=key, interp=interp), A, B, N, px, py)
    plt.savefig(os.path.join(savedir, f'trajectories_{interp_name}-{m}-{n}.png'))



if __name__ == '__main__':
    fire.Fire(main)