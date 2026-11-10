# flow

Integrate learned vector fields: sampling with ODEs (`Flow`) and SDEs
(`Process`), and exact log-densities by the instantaneous change of variables.
`flow` does not import `sinterp`; any callable `v(x, t)` works, including the
`get_velocity_b` / `get_score_s` methods of a trained `sinterp` field.

![ODE and SDE sampling and push-forward density](../docs/images/flow/flow_2d.png?raw=true)

*The exact LinearDeterministic field (computed in closed form, as in
`sinterp.exact_si_gmm`), integrated from N(0, I) to an 8-mode mixture. Left:
`Flow.forward_trajectory`. Middle: `Process.b_forward_trajectory`. Right:
density from `Flow.push_backward_log_prob`, against the true target density.*

## Conventions

- Time runs from `t = 0` (source, `p_0`) to `t = 1` (target, `p_1`), the same
  convention as `sinterp`, where `x_0` is the source and `x_1` the data.
- A vector field is called as `v(x, t)` with a single (unbatched) state `x`,
  which may be an array or a PyTree (e.g. a dict of blocks), and a scalar `t`.
- `forward`, `backward`, `push_forward`, ... act on a single sample; the
  `b_`-prefixed methods are batched (`vmap`) and JIT-compiled.

## `Flow` (ODE)

```python
flow.Flow(velocity, n_steps, integrator_name="rk4", **integrator_kwargs)
```

Solves `dx/dt = v(x, t)` on a fixed grid of `n_steps` steps.

| method | returns |
| :--- | :--- |
| `forward(x0)` / `backward(x1)` | the end point; `backward` integrates `dx/ds = -v(x, 1 - s)` |
| `forward_trajectory(x0)` / `backward_trajectory(x1)` | the `n_steps + 1` states, including the start |
| `push_forward_log_prob(logp_x0, x0, key=None)` | `(log p_1(x_1), x_1)`, with `log p_1(x_1) = log p_0(x_0) - int_0^1 div v dt` |
| `push_backward_log_prob(logp_x1, x1, key=None)` | `(log p_0(x_0), x_0)`, i.e. `logp_x1 + int_0^1 div v dt` |
| `push_forward(p_x0, x0)` / `push_backward(p_x1, x1)` | the same in probability space |
| `b_forward`, `b_backward`, `b_*_trajectory`, `b_push_*` | batched, JIT-compiled versions |

The divergence integral uses the trapezoid rule on the trajectory grid (second order
in the step size, one more divergence evaluation than steps). It uses the
exact divergence (`divergence_jvp`, one JVP per dimension) by default, or the
Hutchinson estimator (`divergence_hutchinson`, one JVP with a Gaussian probe,
fresh at every step) when a `key` is passed.

To evaluate the target density at a point `x`, integrate backwards and use the
base density:

```python
int_div, x0 = f.push_backward_log_prob(0.0, x)
log_p1 = log_p0(x0) - int_div
```

The batched entry points are `jax.jit`-decorated methods of an `eqx.Module`, so
the velocity must be a PyTree: an `eqx.Module` (a network, or a bound method of
a `sinterp` field) works; a bare Python `lambda` does not. With a lambda, use
`jax.jit(jax.vmap(f.forward))` instead.

## `Process` (SDE)

```python
flow.Process(b, s, epsilon, n_steps, integrator_name="euler-maruyama")
```

Given the velocity `b(x, t)`, the score `s(x, t)` and a diffusion coefficient
`epsilon(t) >= 0`, simulates

- forward: `dX = [b(X, t) + epsilon(t) s(X, t)] dt + sqrt(2 epsilon(t)) dW`,
- backward, in reversed time `tau = 1 - t`:
  `dY = -[b(Y, 1 - tau) - epsilon(1 - tau) s(Y, 1 - tau)] dtau + sqrt(2 epsilon(1 - tau)) dW`,

whose marginals equal those of the ODE `dx/dt = b` for every choice of
`epsilon` ([Albergo, Boffi & Vanden-Eijnden](https://arxiv.org/abs/2303.08797),
eqs. 2.21 and 2.23). Methods take a PRNG key: `forward(x0, key)`,
`forward_trajectory(x0, key)`, `b_forward(x0s, keys)`, ... `Process` supports
sampling only, not densities.

## Integrators

`get_integrator(name)` returns the integrator class for `'euler'`, `'rk4'`,
`'euler-maruyama'` (SDEs; needs a diffusion function), or `'diffrax'` (adaptive
Dopri5 with a PID controller, rtol = atol = 1e-5; `'diffrax-kvaerno5'` for a
stiff solver). With diffrax, `n_steps` only sets the initial step size, and
`DiffraxIntegrator.solve_with_stats` also returns the number of steps taken.

## Networks

`flow.net` has small Equinox models with a Fourier-feature time embedding:
`VelocityNet(dim, key, depth=4, width=256)` (output layer initialised to zero,
so the initial field is 0), `DualOutputNet` and `EtaNet`. They take a single
`(x, t)` with `x` of shape `(dim,)`.

## Example

```python
import jax
import jax.numpy as jnp
import flow

# A closed-form field so the snippet runs as-is; in practice this is a trained
# network, e.g. a sinterp field's get_velocity_b.
velocity = lambda x, t: -0.5 * x
f = flow.Flow(velocity, n_steps=50)

x0 = jax.random.normal(jax.random.PRNGKey(0), (2,))      # x0 ~ p_0 = N(0, I)
log_p0 = jax.scipy.stats.multivariate_normal.logpdf(x0, jnp.zeros(2), jnp.eye(2))

x1 = f.forward(x0)                                       # a sample of p_1
log_p1, x1 = f.push_forward_log_prob(log_p0, x0)         # and its log-density
# here p_1 = N(0, e^-1 I), so this matches the closed form:
print(log_p1, jax.scipy.stats.multivariate_normal.logpdf(x1, jnp.zeros(2), jnp.exp(-1.0) * jnp.eye(2)))

xs = jax.jit(jax.vmap(f.forward))(jax.random.normal(jax.random.PRNGKey(1), (1000, 2)))
```
