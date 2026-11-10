# sinterp

[Stochastic interpolants](https://arxiv.org/abs/2303.08797) and continuous
normalizing flows in [JAX](https://github.com/google/jax) /
[Equinox](https://github.com/patrick-kidger/equinox).

The distribution `sinterp-jax` installs two independent packages:

| package | what it does | docs |
| :--- | :--- | :--- |
| `sinterp` | **what to learn.** Interpolants $x_t = \alpha(t)\,x_0 + \beta(t)\,x_1 + \gamma(t)\,z$ between a source $x_0$ and data $x_1$; the stochastic fields (velocity $b$, score $s$, denoisers $\eta$) parameterised in several ways, with their regression losses; noise-data couplings (independent, minibatch OT, condition-aware OT, per-block Knothe-Rosenblatt); exact fields for Gaussian and Gaussian-mixture endpoints, for testing. | [`sinterp/README.md`](sinterp/README.md) |
| `flow` | **how to use what was learned.** Integrates a vector field $v(x, t)$ from $t = 0$ to $t = 1$ as an ODE (`Flow`) or SDE (`Process`), forwards or backwards, and computes exact log-densities by the instantaneous change of variables (exact or Hutchinson divergence). | [`flow/README.md`](flow/README.md) |

They fit together as a training / sampling pair: build a `sinterp` field, train
it with its loss on (coupled) noise-data pairs, then hand its
`get_velocity_b` (and, for an SDE, `get_score_s`) to `flow`. Neither package
imports the other, so `flow` can integrate any field and `sinterp` fields can
be sampled with any solver. Both use the same time convention: $t = 0$ is the
source (noise), $t = 1$ the data.

## Install

Install JAX for your hardware first (`pip install "jax[cpu]"` or
`"jax[cuda12]"`), then

```bash
pip install "sinterp-jax @ git+https://github.com/act65/sinterp@v0.3.0"
```

```python
import sinterp, flow
```

From a clone: `pip install -e ".[dev]"` and `pytest`.

## Quickstart

Learn a one-sided flow from N(0, I) to a 2-D mixture, then sample it and
evaluate log-densities. Runs in a few seconds on a CPU.

```python
import jax, jax.numpy as jnp, equinox as eqx
import sinterp, flow

def sample_data(key, n):                       # 4-mode mixture in 2-D
    k1, k2 = jax.random.split(key)
    centres = jnp.array([[2., 2.], [-2., 2.], [-2., -2.], [2., -2.]])
    return centres[jax.random.randint(k1, (n,), 0, 4)] + 0.3 * jax.random.normal(k2, (n, 2))

interp = sinterp.OneSidedLinear()              # x_t = (1 - t) z + t x1
field = sinterp.OneSidedField_b(interp, flow.VelocityNet(dim=2, key=jax.random.PRNGKey(0), depth=2, width=64))

def loss_fn(field, key):
    kx, kz, kt, kc = jax.random.split(key, 4)
    x1 = sample_data(kx, 256)
    z = jax.random.normal(kz, x1.shape)
    z, x1, *_ = sinterp.IndependentCoupling()(kc, z, x1)   # or an OT coupling
    t = jax.random.uniform(kt, (256,))
    # loss_b(x0, x1, z, t) regresses b(t, x_t) onto d/dt x_t; x0 is unused when one-sided
    return jax.vmap(field.loss_b)(z, x1, z, t).mean()

@eqx.filter_jit
def step(field, key, lr=3e-3):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(field, key)
    return eqx.apply_updates(field, jax.tree_util.tree_map(lambda g: -lr * g, grads)), loss

key = jax.random.PRNGKey(1)
for i in range(2000):
    key, sub = jax.random.split(key)
    field, loss = step(field, sub)

# sample: integrate the learned velocity with flow (RK4, 50 steps)
f = flow.Flow(field.get_velocity_b, n_steps=50)
z = jax.random.normal(jax.random.PRNGKey(2), (1000, 2))
x1 = f.b_forward(z)
# exact log-density of the samples under the learned flow
log_p0 = jax.scipy.stats.multivariate_normal.logpdf(z, jnp.zeros(2), jnp.eye(2))
log_p1, _ = f.b_push_forward_log_prob(log_p0, z)
```

For a two-sided interpolant (a learned bridge between two data distributions)
use e.g. `sinterp.LinearVPStochastic()` with `sinterp.TwoSidedField_v_etaz`, and
sample with `flow.Process(field.get_velocity_b, field.get_score_s, epsilon, n_steps)`
when an SDE is wanted. [`sinterp/README.md`](sinterp/README.md) explains which
field class can return which quantity for which interpolant.

## Gallery

All figures are generated from scratch by
[`docs/make_figures.py`](docs/make_figures.py) using exact (closed-form)
Gaussian-mixture fields, so they show what each object *should* learn.

**Interpolant schedules** (`sinterp.interpolants`): the weights on the source
$x_0$, data $x_1$ and noise $z$.

![interpolant coefficients](docs/images/interpolants/coefficients.png?raw=true)

**Density and probability-flow ODE paths** of every interpolant, from a 3-mode
to a 2-mode mixture in 1-D (paths integrated with `flow.Flow`):

![density and ODE paths per interpolant](docs/images/pbs/overview.png?raw=true)

**The exact fields of one interpolant**: density $p$, velocity $b$, score $s$,
$v$, and ODE / SDE samples (`flow.Flow`, `flow.Process`). Other interpolants are
in [`docs/images/pbs/`](docs/images/pbs/).

![p, b, s, v for LinearStochastic](docs/images/pbs/LinearStochastic.png?raw=true)

**Couplings** (`sinterp.couplings`): which data point each noise sample is
trained against.

![couplings](docs/images/couplings/unconditional.png?raw=true)

Joint OT couplings make the training noise of a child block depend on its
parent; per-block C2OT (`C2OTCoupling.couple_blocks`) keeps it independent:

![joint vs per-block couplings](docs/images/couplings/blocks.png?raw=true)

**flow**: ODE and SDE sampling and push-forward densities in 2-D:

![flow](docs/images/flow/flow_2d.png?raw=true)

More in [`sinterp/README.md`](sinterp/README.md) (stochastic-field
parameterisations, conditional couplings) and [`flow/README.md`](flow/README.md).

## Repository layout

```
sinterp/            interpolants, stochastic fields, couplings, exact references
flow/               ODE / SDE integration, densities, small networks
docs/make_figures.py  regenerates docs/images/ (JAX_PLATFORMS=cpu python docs/make_figures.py)
docs/notebooks/     an interactive notebook of the interpolant schedules
```

## License

MIT, see [LICENSE](LICENSE).
