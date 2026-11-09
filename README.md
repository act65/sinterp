# sinterp

[Stochastic interpolants](https://arxiv.org/abs/2303.08797) in
[JAX](https://github.com/google/jax) / [Equinox](https://github.com/patrick-kidger/equinox):
interpolants, noise–data couplings, the stochastic fields (velocity, drift,
score, denoisers) and their losses, plus exact Gaussian / Gaussian-mixture
references for testing.

| module | contents |
| :--- | :--- |
| `sinterp.interpolants` | one- and two-sided interpolants (linear, trigonometric, encoding–decoding, ...) |
| `sinterp.stochasticfield` | velocity / drift / score / $\eta$-prediction fields and their training losses |
| `sinterp.field_factory` | build fields for array- and dict-valued (block) data |
| `sinterp.couplings` | noise–data couplings, below |
| `sinterp.exact_si_gaussian`, `exact_si_gmm`, `exact_ot_gmm` | closed-form interpolant fields for Gaussian and GMM targets |

## Couplings

- `IndependentCoupling`: $p(x_0, x_1) = p(x_0)\,p(x_1)$.
- `SinkhornCoupling`: minibatch entropic OT
  ([Multisample Flow Matching](https://proceedings.mlr.press/v202/pooladian23a.html),
  [Tong et al. 2023](https://arxiv.org/abs/2302.00482)); fixed-iteration
  log-domain Sinkhorn by default.
- `ExactOTCoupling` / `ExactConditionalCoupling`: minibatch exact OT (a
  Hungarian permutation, so every data row keeps one real noise draw).
- `C2OTCoupling`: condition-aware OT
  ([Cheng & Schwing, ICCV 2025](https://arxiv.org/abs/2503.10636)), with the
  adaptive condition weight computed in closed form. `C2OTCoupling(w=0.0)` is
  plain Hungarian matching. `couple_blocks` couples dict-valued data one block
  at a time in topological order, with each block's parents as the condition: a
  minibatch Knothe–Rosenblatt coupling, which is what makes block-triangular
  ("dynamic") conditional flows consistent (see
  [structflow](https://github.com/act65/structflow)).

Coupling a full vector with OT is not neutral for *conditional* models: the
noise paired with a block then depends on that block's conditions. Use
independent pairing or `C2OTCoupling` when a model is conditioned on part of
its input.

See [`sinterp/README.md`](sinterp/README.md) for the mathematics, and `play/`
for plots of the interpolants.

![Const noise](play/viz-interpolators/ConstantNoise.png?raw=true)
![LS](play/viz-interpolators/LinearStochastic.png?raw=true)
![TS](play/viz-interpolators/TrigonometricStochastic.png?raw=true)

## Install

Install JAX for your hardware first (`pip install "jax[cpu]"` or
`"jax[cuda12]"`), then

```bash
pip install "sinterp-jax @ git+https://github.com/act65/sinterp@v0.2.0"   # import sinterp
```

or, from a clone, `pip install -e ".[dev]"` and run `pytest`.

## License

GPL-3.0, see [LICENSE](LICENSE).
