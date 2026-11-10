"""Regenerate every figure used in the READMEs.

    JAX_PLATFORMS=cpu python docs/make_figures.py            # all figures
    JAX_PLATFORMS=cpu python docs/make_figures.py couplings  # one group

Groups: interpolants, pbs, fields, couplings, flow. Figures are written to
docs/images/<group>/. Everything is computed from scratch with ``sinterp``,
``flow``, numpy, jax and matplotlib: the fields are the exact (closed-form)
fields of the stochastic interpolant between Gaussian mixtures
(``sinterp.exact_si_gmm``), so no network is trained.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from PIL import Image

import flow
from sinterp import interpolants as I
from sinterp import stochasticfield as SF
from sinterp.couplings import (
    IndependentCoupling,
    SinkhornCoupling,
    ExactOTCoupling,
    C2OTCoupling,
)
from sinterp.exact_si_gmm import construct_p_b_s
from sinterp.utils import get_v

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
DPI = 150
TAB = plt.get_cmap("tab10").colors
C_ALPHA, C_BETA, C_GAMMA = TAB[0], TAB[1], TAB[2]
SEQ = "viridis"
DIV = "RdBu_r"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})


def save(fig, group, name):
    d = os.path.join(OUT, group)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    fig.savefig(path)
    plt.close(fig)
    # 256-colour palette PNG: about 3x smaller, visually identical for these plots
    Image.open(path).convert("RGB").quantize(colors=256, dither=Image.Dither.NONE).save(path, optimize=True)
    print("wrote", os.path.relpath(path, os.path.dirname(HERE)))


# ---------------------------------------------------------------------------
# Toy distributions and exact fields
# ---------------------------------------------------------------------------

def gmm(means, var, weights=None):
    """A Gaussian mixture as (means (K, d), covs (K, d, d), weights (K,))."""
    means = jnp.asarray(means, dtype=jnp.float32)
    k, d = means.shape
    covs = jnp.stack([var * jnp.eye(d)] * k)
    w = jnp.full((k,), 1.0 / k) if weights is None else jnp.asarray(weights)
    return means, covs, w


def many_g(n, m, var):
    """Two 1-D equal-weight mixtures with n and m components at linspace(-k/2, k/2, k).

    Local stand-in for ``exp_utils.data.gaussians.many_g`` used by the original plotting
    script. That script called ``many_g(3, 2, std=0.1)`` and passed each component's ``std``
    to ``exact_si_gmm`` where a covariance is expected, so its fields were those of
    components with *variance* 0.1; ``var=0.1`` reproduces those plots."""
    def lin(k):
        return gmm(np.linspace(-k / 2, k / 2, k)[:, None], var)
    return lin(n), lin(m)


# 1-D example: 3 equal modes at (-1.5, 0, 1.5) -> 2 equal modes at (-1, 1), component variance 0.1.
SRC_1D, TGT_1D = many_g(3, 2, var=0.1)
NOISE_1D = gmm([[0.0]], 1.0)  # stands in for x0 with one-sided interpolants (where x0 is unused)


def gmm_sample(key, dist, n):
    means, covs, w = dist
    k1, k2 = jax.random.split(key)
    idx = jax.random.choice(k1, means.shape[0], (n,), p=w)
    L = jnp.linalg.cholesky(covs)[idx]
    return means[idx] + jnp.einsum("nij,nj->ni", L, jax.random.normal(k2, (n, means.shape[1])))


def gmm_logpdf(x, dist):
    means, covs, w = dist
    lp = jax.vmap(lambda m, c: jax.scipy.stats.multivariate_normal.logpdf(x, m, c))(means, covs)
    return jax.scipy.special.logsumexp(lp + jnp.log(w))


def is_one_sided(interp):
    return all(float(interp.alpha(t)) == 0.0 for t in (0.1, 0.5, 0.9))


def exact_fields(interp, src, tgt):
    """Exact fields of x_t = alpha x0 + beta x1 + gamma z with x0 ~ src, x1 ~ tgt, z ~ N(0, I)
    independent. p, b, s come from sinterp.exact_si_gmm and v from sinterp.utils.get_v; the
    denoisers are computed here from the Gaussian posterior of each pair (i, j) of mixture
    components, e.g. E[x0 | x_t = x, ij] = m0_i + alpha C0_i C_t^-1 (x - m_t).
    Every callable takes (x of shape (d,), scalar t)."""
    m0, c0, w0 = src
    m1, c1, w1 = tgt
    p, b, s = construct_p_b_s(interp, list(m0), list(m1), list(c0), list(c1), list(w0), list(w1))
    v = get_v(b, s, interp)
    d = m0.shape[1]
    mvn = jax.vmap(jax.vmap(jax.scipy.stats.multivariate_normal.logpdf, (None, 0, 0)), (None, 0, 0))

    def denoisers(x, t):
        a, bb, g = interp.alpha(t), interp.beta(t), interp.gamma(t)
        M = a * m0[:, None] + bb * m1[None, :]                                # (n, m, d)
        C = a ** 2 * c0[:, None] + bb ** 2 * c1[None, :] + g ** 2 * jnp.eye(d)
        r = jnp.linalg.solve(C, (x - M)[..., None])[..., 0]                   # C^-1 (x - m)
        lw = jnp.log(w0)[:, None] + jnp.log(w1)[None, :] + mvn(x, M, C)
        pi = jax.nn.softmax(lw.reshape(-1)).reshape(lw.shape)[..., None]
        e0 = jnp.sum(pi * (m0[:, None] + a * jnp.einsum("nij,nmj->nmi", c0, r)), (0, 1))
        e1 = jnp.sum(pi * (m1[None, :] + bb * jnp.einsum("mij,nmj->nmi", c1, r)), (0, 1))
        ez = jnp.sum(pi * g * r, (0, 1))
        return e0, e1, ez, -jnp.sum(pi * r, (0, 1))

    def b_logdomain(x, t):
        e0, e1, ez, _ = denoisers(x, t)
        return interp.dalphadt(t) * e0 + interp.dbetadt(t) * e1 + interp.dgammadt(t) * ez

    # b_ld / s_ld: the same b and s, but with the mixture posterior computed in log space.
    # sinterp.exact_si_gmm divides raw densities, which underflow (0/0 = nan) far from all
    # modes in float32; the 2-D flow figure integrates through such regions.
    return dict(p=p, b=b, s=s, v=v,
                eta0=lambda x, t: denoisers(x, t)[0],
                eta1=lambda x, t: denoisers(x, t)[1],
                etaz=lambda x, t: denoisers(x, t)[2],
                b_ld=b_logdomain,
                s_ld=lambda x, t: denoisers(x, t)[3])


def grid_eval(fn, ts, xs):
    """A 1-D field fn(x (1,), t) on a (len(ts), len(xs)) grid."""
    f = jax.jit(jax.vmap(jax.vmap(lambda x, t: jnp.reshape(fn(x[None], t), ()), (0, None)), (None, 0)))
    return np.asarray(f(xs, ts))


def clipped(fn, eps=2e-3):
    """Clip t away from 0 and 1, where some schedules have infinite derivatives."""
    return lambda x, t: fn(x, jnp.clip(t, eps, 1 - eps))


def sample_x_at(interp, key, src, tgt, n, t):
    k0, k1, kz = jax.random.split(key, 3)
    x0, x1 = gmm_sample(k0, src, n), gmm_sample(k1, tgt, n)
    return interp(x0, x1, jax.random.normal(kz, x0.shape), t)


ALL_INTERPS = [
    ("LinearDeterministic", I.LinearDeterministic()),
    ("SquaredDeterministic", I.SquaredDeterministic()),
    ("ConstantNoise (sigma=1e-3)", I.ConstantNoise()),
    ("LinearStochastic", I.LinearStochastic()),
    ("SquaredStochastic", I.SquaredStochastic()),
    ("TrigonometricStochastic", I.TrigonometricStochastic()),
    ("LinearVPStochastic", I.LinearVPStochastic()),
    ("TrigonometricVPStochastic (sigma=0.5)", I.TrigonometricVPStochastic()),
    ("EncodingDecodingStochastic", I.EncodingDecodingStochastic()),
    ("LowNoiseEncodingDecodingStochastic", I.LowNoiseEncodingDecodingStochastic()),
    ("OneSidedLinear", I.OneSidedLinear()),
    ("OneSidedSBDM", I.OneSidedSBDM()),
]


def short(name):
    return name.split(" (")[0]


def check_exact_fields():
    """b from sinterp.exact_si_gmm must equal alpha' eta0 + beta' eta1 + gamma' etaz."""
    for name, it in ALL_INTERPS:
        F = exact_fields(it, NOISE_1D if is_one_sided(it) else SRC_1D, TGT_1D)
        for t in (0.13, 0.41, 0.77):
            for x in (-1.1, 0.3, 0.9):
                xx = jnp.array([x])
                b = F["b"](xx, t)
                b2 = (it.dalphadt(t) * F["eta0"](xx, t) + it.dbetadt(t) * F["eta1"](xx, t)
                      + it.dgammadt(t) * F["etaz"](xx, t))
                assert jnp.allclose(b, b2, rtol=1e-3, atol=1e-3), (name, t, x, b, b2)
                assert jnp.allclose(b, F["b_ld"](xx, t), rtol=1e-3, atol=1e-3), (name, t, x)
                assert jnp.allclose(F["s"](xx, t), F["s_ld"](xx, t), rtol=1e-3, atol=1e-3), (name, t, x)


# ---------------------------------------------------------------------------
# 1. Interpolant gallery
# ---------------------------------------------------------------------------

def fig_interpolants():
    ts = jnp.linspace(0.0, 1.0, 401)
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.6), sharex=True, sharey=True)
    for ax, (name, it) in zip(axes.flat, ALL_INTERPS):
        a, b, g = (np.asarray(jax.vmap(f)(ts)) for f in (it.alpha, it.beta, it.gamma))
        ax.plot(ts, a, color=C_ALPHA, lw=2, label=r"$\alpha(t)$, weight on $x_0$")
        ax.plot(ts, b, color=C_BETA, lw=2, label=r"$\beta(t)$, weight on $x_1$")
        ax.plot(ts, g, color=C_GAMMA, lw=2, label=r"$\gamma(t)$, weight on $z$")
        ax.plot(ts, a ** 2 + b ** 2 + g ** 2, color="0.45", lw=1.2, ls="--",
                label=r"$\alpha^2+\beta^2+\gamma^2$")
        ax.set_title(name, fontsize=8.5)
        ax.set_ylim(-0.05, 1.12)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("t")
    for ax in axes[:, 0]:
        ax.set_ylabel("coefficient")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle(r"Interpolant schedules: $x_t = \alpha(t)\,x_0 + \beta(t)\,x_1 + \gamma(t)\,z$"
                 r"   (one-sided: $\alpha \equiv 0$ and $z$ is the source)", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "interpolants", "coefficients.png")

    # sample paths: one (x0, x1, z) per path, held fixed over t
    key = jax.random.PRNGKey(0)
    n = 60
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.6), sharex=True, sharey=True)
    for ax, (name, it) in zip(axes.flat, ALL_INTERPS):
        k0, k1, kz = jax.random.split(key, 3)
        x0, x1 = gmm_sample(k0, SRC_1D, n), gmm_sample(k1, TGT_1D, n)
        z = jax.random.normal(kz, (n, 1))
        paths = np.asarray(jax.vmap(lambda t: it(x0, x1, z, t))(ts))[..., 0]  # (T, n)
        ax.plot(ts, paths, color=TAB[0], lw=0.6, alpha=0.45)
        ax.set_title(name, fontsize=8.5)
        ax.set_ylim(-3.2, 3.2)
    for ax in axes[-1]:
        ax.set_xlabel("t")
    for ax in axes[:, 0]:
        ax.set_ylabel("$x_t$")
    fig.suptitle("Sample paths $x_t$ for fixed $(x_0, x_1, z)$, 3-mode to 2-mode 1-D mixture "
                 "(one-sided: from $z \\sim N(0,1)$)", y=0.995)
    fig.tight_layout()
    save(fig, "interpolants", "paths.png")


# ---------------------------------------------------------------------------
# 2. Exact p, b, s, v and ODE/SDE samples (the "pbs" plots)
# ---------------------------------------------------------------------------

XLIM = 3.0
TS = jnp.linspace(0.005, 0.995, 180)
XS = jnp.linspace(-XLIM, XLIM, 200)
EXTENT = [0, 1, -XLIM, XLIM]


def density_panel(ax, P, title=None, cbar=True):
    im = ax.imshow(P.T, origin="lower", extent=EXTENT, aspect="auto", cmap=SEQ,
                   norm=PowerNorm(0.5, vmin=0, vmax=np.percentile(P, 99.5)))
    if title:
        ax.set_title(title)
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="density (sqrt scale)")
    return im


def signed_image(ax, F, P, ref=None):
    """Signed field F on (t, x); grey where the density is negligible."""
    ref = F if ref is None else ref
    mask = P < 2e-3 * P.max()
    lim = np.percentile(np.abs(ref[~mask]), 98) + 1e-9
    ax.set_facecolor("0.85")
    return ax.imshow(np.ma.array(np.clip(F, -lim, lim), mask=mask).T, origin="lower", extent=EXTENT,
                     aspect="auto", cmap=DIV, norm=TwoSlopeNorm(0.0, -lim, lim))


def signed_panel(ax, F, P, title, label):
    im = signed_image(ax, F, P)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=label)


def ode_sde_paths(interp, F, src, tgt, n=60, n_steps=200, eps=0.3):
    x0 = sample_x_at(interp, jax.random.PRNGKey(1), src, tgt, n, 0.0)
    ode = flow.Flow(clipped(F["b"]), n_steps=n_steps, integrator_name="rk4")
    ode_paths = jax.jit(jax.vmap(ode.forward_trajectory))(x0)
    sde = flow.Process(clipped(F["b"]), clipped(F["s"]), lambda t: eps, n_steps=n_steps)
    sde_paths = sde.b_forward_trajectory(x0, jax.random.split(jax.random.PRNGKey(2), n))
    tt = np.linspace(0, 1, n_steps + 1)
    return tt, np.asarray(ode_paths)[..., 0].T, np.asarray(sde_paths)[..., 0].T


PBS_INTERPS = [
    ("LinearDeterministic", I.LinearDeterministic()),
    ("LinearStochastic", I.LinearStochastic()),
    ("TrigonometricStochastic", I.TrigonometricStochastic()),
    ("LinearVPStochastic", I.LinearVPStochastic()),
    ("EncodingDecodingStochastic", I.EncodingDecodingStochastic()),
    ("OneSidedLinear", I.OneSidedLinear()),
]


def fig_pbs():
    for name, it in PBS_INTERPS:
        src = NOISE_1D if is_one_sided(it) else SRC_1D
        F = exact_fields(it, src, TGT_1D)
        P = grid_eval(F["p"], TS, XS)
        B, S, V = (grid_eval(clipped(F[k]), TS, XS) for k in "bsv")
        tt, ode, sde = ode_sde_paths(it, F, src, TGT_1D)

        fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), sharex=True, sharey=True)
        density_panel(axes[0, 0], P, "p(t, x): density of $x_t$")
        signed_panel(axes[0, 1], B, P, r"b(t, x) = E[$\dot x_t$ | $x_t$ = x]  (ODE velocity)", "b")
        signed_panel(axes[0, 2], S, P, r"s(t, x) = $\nabla_x \log p(t, x)$  (score)", "s")
        signed_panel(axes[1, 0], V, P, r"v(t, x) = b + $\gamma\dot\gamma$ s = E[$\dot\alpha x_0 + \dot\beta x_1$ | $x_t$ = x]", "v")
        density_panel(axes[1, 1], P, cbar=False)
        axes[1, 1].plot(tt, ode, color="white", lw=0.6, alpha=0.8)
        axes[1, 1].set_title("ODE  dX = b dt  (flow.Flow, RK4)")
        density_panel(axes[1, 2], P, cbar=False)
        axes[1, 2].plot(tt, sde, color="white", lw=0.5, alpha=0.7)
        axes[1, 2].set_title(r"SDE  dX = (b + $\epsilon$ s) dt + $\sqrt{2\epsilon}$ dW, $\epsilon$ = 0.3 (flow.Process)")
        for ax in axes[1]:
            ax.set_xlabel("t")
        for ax in axes[:, 0]:
            ax.set_ylabel("x")
        for ax in axes.flat:
            ax.set_ylim(-XLIM, XLIM)
        srcname = "N(0, 1)" if is_one_sided(it) else "3-mode mixture"
        fig.suptitle(f"{name}: exact fields from {srcname} (t = 0) to 2-mode mixture (t = 1); "
                     "grey = negligible density", y=0.995)
        fig.tight_layout()
        save(fig, "pbs", f"{name}.png")

    # overview: density + ODE paths for every interpolant
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.6), sharex=True, sharey=True)
    for ax, (name, it) in zip(axes.flat, ALL_INTERPS):
        src = NOISE_1D if is_one_sided(it) else SRC_1D
        F = exact_fields(it, src, TGT_1D)
        P = grid_eval(F["p"], TS, XS)
        tt, ode, _ = ode_sde_paths(it, F, src, TGT_1D, n=40, n_steps=150)
        density_panel(ax, P, cbar=False)
        ax.plot(tt, ode, color="white", lw=0.5, alpha=0.75)
        ax.set_title(name, fontsize=8.5)
        ax.set_ylim(-XLIM, XLIM)
    for ax in axes[-1]:
        ax.set_xlabel("t")
    for ax in axes[:, 0]:
        ax.set_ylabel("x")
    fig.suptitle("Density of $x_t$ (sqrt scale) and probability-flow ODE paths (white) for each interpolant",
                 y=0.995)
    fig.tight_layout()
    save(fig, "pbs", "overview.png")


# ---------------------------------------------------------------------------
# 3. Stochastic-field parameterisations
# ---------------------------------------------------------------------------

QLABEL = {"b": "b", "v": "v", "s": "s", "eta0": r"$\eta_0$ = E[$x_0$ | x]",
          "eta1": r"$\eta_1$ = E[$x_1$ | x]", "etaz": r"$\eta_z$ = E[z | x]"}
GETTER = {"b": "get_velocity_b", "v": "get_velocity_v", "s": "get_score_s",
          "eta0": "get_denoiser_eta0", "eta1": "get_denoiser_eta1", "etaz": "get_denoiser_etaz"}
TWO = ["TwoSidedField_v_etaz", "TwoSidedField_b_s", "TwoSidedField_x0_x1", "TwoSidedField_b_eta1"]
ONE = ["OneSidedField_b", "OneSidedField_eta1"]


def make_fields(it, F):
    """Each StochasticField class, with its networks replaced by the exact quantities it learns."""
    return {
        "TwoSidedField_v_etaz": (SF.TwoSidedField_v_etaz(it, F["v"], F["etaz"]), ("v", "etaz")),
        "TwoSidedField_b_s": (SF.TwoSidedField_b_s(it, F["b"], F["s"]), ("b", "s")),
        "TwoSidedField_x0_x1": (SF.TwoSidedField_x0_x1(it, F["eta0"], F["eta1"]), ("eta0", "eta1")),
        "TwoSidedField_b_eta1": (SF.TwoSidedField_b_eta1(it, F["b"], F["eta1"]), ("b", "eta1")),
        "OneSidedField_b": (SF.OneSidedField_b(it, F["b"]), ("b",)),
        "OneSidedField_eta1": (SF.OneSidedField_eta1(it, F["eta1"]), ("eta1",)),
    }


def field_grid(rows, it, src, quants, fname, title):
    F = exact_fields(it, src, TGT_1D)
    fields = make_fields(it, F)
    P = grid_eval(F["p"], TS, XS)
    refs = {q: grid_eval(F[q], TS, XS) for q in quants}
    fig, axes = plt.subplots(len(rows), len(quants),
                             figsize=(2.1 * len(quants) + 0.8, 1.75 * len(rows) + 1.0),
                             sharex=True, sharey=True, squeeze=False)
    for r, cls in enumerate(rows):
        field, learned = fields[cls]
        for c, q in enumerate(quants):
            ax = axes[r, c]
            signed_image(ax, grid_eval(getattr(field, GETTER[q]), TS, XS), P, ref=refs[q])
            ax.set_yticks([-2, 0, 2])
            tag = "learned" if q in learned else "derived"
            if tag == "learned":
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_color(TAB[1])
                    s.set_linewidth(2.5)
            if r == 0:
                ax.set_title(QLABEL[q], fontsize=9)
            ax.text(0.03, 0.95, tag, transform=ax.transAxes, va="top", fontsize=7.5,
                    color="black", fontweight="bold" if tag == "learned" else "normal",
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
        axes[r, 0].set_ylabel(cls.replace("Field_", "Field_\n") + "\n\nx", fontsize=8.5)
    for ax in axes[-1]:
        ax.set_xlabel("t")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    save(fig, "fields", fname)


def rel_err(G, ref, P):
    return float(np.sqrt(np.sum(P * (G - ref) ** 2) / (np.sum(P * ref ** 2) + 1e-30)))


def fig_fields():
    field_grid(TWO, I.LinearVPStochastic(), SRC_1D, ["b", "v", "s", "eta0", "eta1", "etaz"], "two_sided.png",
               "Two-sided fields, LinearVPStochastic, 3-mode to 2-mode mixture. Each row learns two quantities "
               "(orange frame) and derives the rest;\nwith exact inputs every row gives the same fields. "
               "Axes: t horizontal, x vertical; red > 0 > blue; grey = negligible density.")
    field_grid(ONE, I.OneSidedLinear(), NOISE_1D, ["b", "v", "s", "eta1", "etaz"], "one_sided.png",
               "One-sided fields, OneSidedLinear ($x_t$ = (1-t) z + t $x_1$), N(0, 1) to 2-mode mixture.\n"
               r"One learned quantity (orange frame); $\eta_0 \equiv 0$.")

    # which parameterisation can return b and s, per interpolant
    ts = jnp.linspace(0.02, 0.98, 50)
    xs = jnp.linspace(-XLIM, XLIM, 80)
    rows = TWO + ONE
    errs = {q: np.full((len(rows), len(ALL_INTERPS)), np.nan) for q in ("b", "s")}
    for c, (name, it) in enumerate(ALL_INTERPS):
        one = is_one_sided(it)
        F = exact_fields(it, NOISE_1D if one else SRC_1D, TGT_1D)
        P = grid_eval(F["p"], ts, xs)
        refs = {q: grid_eval(F[q], ts, xs) for q in ("b", "s")}
        for r, (cls, (field, _)) in enumerate(make_fields(it, F).items()):
            if cls.startswith("OneSided") and not one:
                continue
            for q in ("b", "s"):
                errs[q][r, c] = rel_err(grid_eval(getattr(field, GETTER[q]), ts, xs), refs[q], P)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.88")
    for ax, q in zip(axes, ("b", "s")):
        E = np.clip(errs[q], 0, 1)
        ax.imshow(np.ma.masked_invalid(E), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for (r, c), e in np.ndenumerate(errs[q]):
            txt = "n/a" if np.isnan(e) else ("0" if e < 5e-3 else f"{min(e, 9.99):.2f}")
            ax.text(c, r, txt, ha="center", va="center", fontsize=7.5,
                    color="black" if (np.isnan(e) or e > 0.6) else "white")
        ax.set_xticks(range(len(ALL_INTERPS)))
        ax.set_xticklabels([short(n) for n, _ in ALL_INTERPS], rotation=55, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=8)
        ax.set_title(f"relative error of the derived {q}  (0 = exact, 1 = returns 0)")
        ax.spines[:].set_visible(False)
    fig.suptitle("Which field class can return b (for the ODE) and s (for the SDE), given exact learned "
                 "quantities?\nDensity-weighted relative error over t in [0.02, 0.98]; "
                 "n/a: one-sided classes assume alpha = 0.", y=1.0)
    fig.tight_layout()
    save(fig, "fields", "recoverability.png")


# ---------------------------------------------------------------------------
# 4. Couplings
# ---------------------------------------------------------------------------

def two_moons(key, n, noise=0.08):
    k1, k2, k3 = jax.random.split(key, 3)
    lab = jax.random.bernoulli(k1, 0.5, (n,)).astype(jnp.int32)
    th = jax.random.uniform(k2, (n,)) * jnp.pi
    upper = jnp.stack([jnp.cos(th), jnp.sin(th)], 1)
    lower = jnp.stack([1 - jnp.cos(th), 0.5 - jnp.sin(th)], 1)
    x = jnp.where(lab[:, None] == 0, upper, lower) + noise * jax.random.normal(k3, (n, 2))
    return (x - jnp.array([0.5, 0.25])) * 1.6, lab


def pair_lines(ax, a, b, colors=None, alpha=0.35):
    for i in range(a.shape[0]):
        ax.plot([a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], lw=0.6, alpha=alpha,
                color="0.45" if colors is None else colors[i])


def style_xy(ax, lim=3.2):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("$x^{(1)}$")
    ax.set_ylabel("$x^{(2)}$")


def fig_couplings():
    n = 128
    kd, kn, kc = jax.random.split(jax.random.PRNGKey(3), 3)
    data, lab = two_moons(kd, n)
    noise = jax.random.normal(kn, (n, 2))

    # --- unconditional couplings. Coupling(key, x0, x1) returns x1 re-paired to the rows of x0.
    # Independent / Sinkhorn / ExactOT are called with (noise, data); C2OT takes (data, noise).
    res = {}
    _, d, *_ = IndependentCoupling()(kc, noise, data)
    res["IndependentCoupling\n(random permutation)"] = (noise, d, False)
    _, d, *_ = SinkhornCoupling(epsilon=0.1)(kc, noise, data)
    res["SinkhornCoupling(epsilon=0.1)\n(barycentric projection)"] = (noise, d, True)
    _, d, *_ = ExactOTCoupling()(kc, noise, data)
    res["ExactOTCoupling()\n(Sinkhorn, epsilon=0.01, barycentric)"] = (noise, d, True)
    _, e, *_ = C2OTCoupling(w=0.0)(kc, data, noise)
    res["C2OTCoupling(w=0), no condition\n(Hungarian: exact OT permutation)"] = (e, data, False)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    for ax, (title, (z, x, bary)) in zip(axes, res.items()):
        z, x = np.asarray(z), np.asarray(x)
        pair_lines(ax, z, x)
        ax.scatter(*np.asarray(noise).T, s=7, color=TAB[0], label="noise $z$", zorder=3)
        ax.scatter(*np.asarray(data).T, s=7, color=TAB[1], label="data $x_1$", zorder=3)
        if bary:
            ax.scatter(*x.T, s=12, marker="x", lw=0.8, color="black", zorder=4,
                       label="paired target (weighted average of data)")
        cost = float(np.mean(np.sum((z - x) ** 2, 1)))
        ax.set_title(f"{title}\nmean $|z - x_1|^2$ = {cost:.2f}")
        style_xy(ax)
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), markerscale=2)
    fig.suptitle(f"Noise-data couplings on a minibatch of {n}: each line joins a noise sample to the "
                 "target it is trained against", y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "couplings", "unconditional.png")

    # --- conditional: condition c = which moon; plain OT vs C2OT
    def cond_pairs(kind, key):
        k1, k2, k3 = jax.random.split(key, 3)
        x, lab = two_moons(k1, n)
        z = jax.random.normal(k2, (n, 2))
        if kind == "independent":
            _, e, *_ = IndependentCoupling()(k3, x, z)
            return x, lab, e, None
        if kind == "ot":
            _, e, *_ = C2OTCoupling(w=0.0)(k3, x, z)
            return x, lab, e, None
        perm, w = C2OTCoupling().matching(x, z, lab[:, None].astype(jnp.float32))  # adaptive w
        return x, lab, z[perm], float(w)

    def gap(e, lab):
        e, lab = np.asarray(e), np.asarray(lab)
        return float(np.linalg.norm(e[lab == 0].mean(0) - e[lab == 1].mean(0)))

    cols = [TAB[0], TAB[3]]
    fig, axes = plt.subplots(2, 3, figsize=(12, 9.0))
    for j, (kind, title) in enumerate([("independent", "IndependentCoupling"),
                                       ("ot", "Hungarian OT, condition ignored\nC2OTCoupling(w=0)"),
                                       ("c2ot", "C2OTCoupling(), adaptive w")]):
        x, lab, e, w = cond_pairs(kind, kd)
        gaps = [gap(*cond_pairs(kind, jax.random.PRNGKey(200 + i))[2:0:-1]) for i in range(20)]
        if w is not None:
            title += (f" = {w:.2g}\n(target_ratio=0.01 is unreachable with two condition\n"
                      "values, so w saturates: same as cond_dist='discrete')" if w > 1e6 else f" = {w:.2f}")
        x, e, labn = np.asarray(x), np.asarray(e), np.asarray(lab)
        labc = [cols[int(i)] for i in labn]
        ax = axes[0, j]
        pair_lines(ax, e, x, labc, alpha=0.4)
        ax.scatter(*x.T, s=8, c=labc, zorder=3)
        ax.scatter(*e.T, s=8, color="0.35", zorder=3)
        ax.set_title(title, fontsize=9)
        style_xy(ax)
        ax = axes[1, j]
        for k in (0, 1):
            sel = labn == k
            ax.scatter(*e[sel].T, s=10, color=cols[k], label=f"noise paired with c = {k}")
            ax.scatter(*e[sel].mean(0), s=150, marker="P", color=cols[k], edgecolor="black", zorder=5)
        ax.set_title("the same noise, coloured by paired condition\n"
                     f"|E[z | c=0] - E[z | c=1]| = {np.mean(gaps):.2f} (mean of 20 batches)")
        style_xy(ax)
    axes[1, 0].legend(loc="lower left", frameon=False)
    fig.suptitle(f"Conditional training pairs, batch {n}: data x (two moons, condition c = moon, coloured), "
                 "noise z (grey).\nAt sampling time z is drawn independently of c, so the noise should not "
                 "separate by condition (bottom row; + marks the mean).", y=1.0)
    fig.tight_layout(h_pad=2.5)
    save(fig, "couplings", "conditional.png")

    # --- blocks: parent a, child b | a. Joint vs per-block (Knothe-Rosenblatt) coupling
    def draw(key, n):
        ka, kb = jax.random.split(key)
        a = jax.random.normal(ka, (n, 1))
        return {"a": a, "b": 0.8 * a + 0.6 * jax.random.normal(kb, (n, 1))}

    def couple(kind, key, n=256):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        x = draw(k1, n)
        e = {"a": jax.random.normal(k2, (n, 1)), "b": jax.random.normal(k4, (n, 1))}
        if kind == "independent":
            _, ec, *_ = IndependentCoupling()(k3, x, e)
            return x["a"], ec["b"]
        if kind == "sinkhorn":   # legacy joint coupling: noise kept in place, data averaged
            _, xc, *_ = SinkhornCoupling(epsilon=0.1)(k3, e, x)
            return xc["a"], e["b"]
        if kind == "joint":
            _, ec, *_ = C2OTCoupling(w=0.0)(k3, x, e)
            return x["a"], ec["b"]
        ec = C2OTCoupling().couple_blocks(k3, x, e, {"a": [], "b": ["a"]})
        return x["a"], ec["b"]

    kinds = [("independent", "IndependentCoupling"),
             ("sinkhorn", "joint SinkhornCoupling(epsilon=0.1)\n(legacy 'joint', barycentric)"),
             ("joint", "joint exact OT on (a, b)\nC2OTCoupling(w=0) on the dict"),
             ("blocks", "per-block C2OT, b conditioned on a\nC2OTCoupling().couple_blocks")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharex=True, sharey=True)
    for ax, (kind, title) in zip(axes, kinds):
        a, eb = couple(kind, jax.random.PRNGKey(10))
        corrs = [float(np.corrcoef(*(np.asarray(v)[:, 0] for v in couple(kind, jax.random.PRNGKey(100 + i))))[0, 1])
                 for i in range(20)]
        ax.scatter(np.asarray(a), np.asarray(eb), s=7, color=TAB[0], alpha=0.8)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_title(f"{title}\ncorr(a, $z_b$) = {np.mean(corrs):+.2f} $\\pm$ {np.std(corrs):.2f} (20 batches)")
        ax.set_xlabel("parent value a")
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
    axes[0].set_ylabel("noise $z_b$ paired with child block b")
    fig.suptitle("Two blocks, a ~ N(0, 1) and b = 0.8 a + 0.6 N(0, 1), batch 256. A flow for b | a is sampled with "
                 "$z_b$ independent of a.\nJoint couplings make the training noise of b depend on its parent; "
                 "the per-block (Knothe-Rosenblatt) coupling does not.", y=1.02)
    fig.tight_layout()
    save(fig, "couplings", "blocks.png")


# ---------------------------------------------------------------------------
# 5. flow: ODE / SDE sampling and push-forward densities in 2-D
# ---------------------------------------------------------------------------

def fig_flow():
    ang = jnp.arange(8) * 2 * jnp.pi / 8
    tgt = gmm(2.2 * jnp.stack([jnp.cos(ang), jnp.sin(ang)], 1), 0.15 ** 2)
    src = gmm([[0.0, 0.0]], 1.0)
    F = exact_fields(I.LinearDeterministic(), src, tgt)
    b = clipped(F["b_ld"])
    n_steps = 60
    f = flow.Flow(b, n_steps=n_steps, integrator_name="rk4")

    z = gmm_sample(jax.random.PRNGKey(0), src, 400)
    traj = np.asarray(jax.jit(jax.vmap(f.forward_trajectory))(z))           # (n, T, 2)
    proc = flow.Process(b, clipped(F["s_ld"]), lambda t: 0.2, n_steps=200)
    straj = np.asarray(proc.b_forward_trajectory(z, jax.random.split(jax.random.PRNGKey(1), 400)))

    L, m = 3.6, 120
    g = jnp.linspace(-L, L, m)
    X = jnp.stack(jnp.meshgrid(g, g), -1).reshape(-1, 2)
    # push_backward_log_prob(0, x) returns (int_0^1 div b dt, x_0), so log p_1(x) = log p_0(x_0) - integral
    acc, x0 = jax.jit(jax.vmap(lambda x: f.push_backward_log_prob(0.0, x)))(X)
    P1 = np.asarray(jnp.exp(jax.vmap(lambda x: gmm_logpdf(x, src))(x0) - acc)).reshape(m, m)
    Pt = np.asarray(jnp.exp(jax.vmap(lambda x: gmm_logpdf(x, tgt))(X))).reshape(m, m)
    print(f"  flow_2d: max |p_flow - p_true| = {np.abs(P1 - Pt).max():.3f} (max p = {Pt.max():.3f})")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    ax = axes[0]
    for k in range(120):
        ax.plot(traj[k, :, 0], traj[k, :, 1], color=TAB[0], lw=0.5, alpha=0.5)
    ax.scatter(*traj[:, 0].T, s=4, color="0.55", label="$x_0$ ~ N(0, I)")
    ax.scatter(*traj[:, -1].T, s=4, color=TAB[1], label="Flow.forward($x_0$)")
    ax.set_title(f"flow.Flow: ODE, RK4, {n_steps} steps")
    ax.legend(loc="lower left", frameon=False, markerscale=2.5)
    ax = axes[1]
    for k in range(120):
        ax.plot(straj[k, :, 0], straj[k, :, 1], color=TAB[2], lw=0.4, alpha=0.45)
    ax.scatter(*straj[:, -1].T, s=4, color=TAB[1])
    ax.set_title(r"flow.Process: SDE, Euler-Maruyama, $\epsilon$ = 0.2")
    ax = axes[2]
    im = ax.imshow(P1, origin="lower", extent=[-L, L, -L, L], cmap=SEQ, norm=PowerNorm(0.5, vmin=0, vmax=P1.max()))
    ax.contour(np.asarray(g), np.asarray(g), Pt, levels=[0.05, 0.3], colors="white", linewidths=0.7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="density (sqrt scale)")
    ax.set_title("density from Flow.push_backward_log_prob\n(white: contours of the true density)")
    for ax in axes:
        style_xy(ax, L)
    fig.suptitle("Integrating the exact LinearDeterministic field (closed form, mixture posterior in log space) "
                 "from N(0, I) to an 8-mode mixture", y=1.0)
    fig.tight_layout()
    save(fig, "flow", "flow_2d.png")


GROUPS = {"interpolants": fig_interpolants, "pbs": fig_pbs, "fields": fig_fields,
          "couplings": fig_couplings, "flow": fig_flow}

if __name__ == "__main__":
    check_exact_fields()
    for name in (sys.argv[1:] or list(GROUPS)):
        GROUPS[name]()
