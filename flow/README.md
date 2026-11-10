### Flow

A example use case is generating samples from a pretrained neural flow.

We support two kinds of dynamical system;

- ODEs (flows)
- SDEs (process aka diffusion models)

The key difference are;

- `flows`;
    - require only a velocity field
    - support `forward` / `backward` evolution (ie sampling), `push_forward` / `push_backward` evolution (allowing the evaluation of p(x))
- `processes`;
    - require a deterministic velocity field (aka the drift) and a score function (aka diffusion)
    - support only `forward` / `backward` evolution (ie sampling)

```python
import jax
import flow
from bayes import Gaussian

# The velocity field v(z, t). In practice this wraps your trained network,
# e.g. `lambda z, t: model(params, z, t)`. Here we use a closed-form field so
# the snippet runs as-is. `n_steps` is fixed when the Flow is constructed.
velocity_field = lambda z, t: -0.5 * z
f = flow.Flow(velocity_field, n_steps=50)

source_distribution = Gaussian(dim=2)        # base p_0 = N(0, I)
key = jax.random.PRNGKey(0)

# sample from p_1
x0 = source_distribution.sample(key)
x1 = f.forward(x0)

# evaluate the probability of an observation x1
x1_observed = x1                             # observations are generated somehow
x0_observed = f.backward(x1_observed)
# requires evaluating p_0(x); easy when p_0 is a Gaussian.
p_x0_observed = source_distribution.prob(x0_observed)
p_x1_observed, _ = f.push_forward(p_x0_observed, x0_observed)
```

> `forward` / `backward` / `push_forward` operate on a single sample. The
> batched, JIT-compiled entry points (`b_forward`, `b_backward`,
> `b_push_forward`, ...) require the velocity field to be an `eqx.Module` (so
> its parameters can be traced and everything else treated as static) — a
> trained network satisfies this, but a bare Python `lambda` does not.