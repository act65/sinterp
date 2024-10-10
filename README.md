The repository provides implementations of [Stochastic interpolants](https://arxiv.org/abs/2209.03003).

Specifically, it provides implementations of;

- the many variants of interpolant
- couplings between the target and source distributions
- various different losses (for the drift, velocity, and score fields)
- an exact implementation of stochastic interpolants for Gaussian distributions and Gaussian mixtures

## Couplings

We implement the;

- independent coupling
- minibatch optimal transport coupling (see [Multisample Flow Matching](http://proceedings.mlr.press/v202/pooladian23a/pooladian23a.pdf) and [Stochastic interpolants with data-dependent couplings](https://arxiv.org/abs/2310.03725) for more details)

## Interpolants

We plot the interpolants defined in `sinterp/interpolants.py`.
Because many of the interpolants require that the coeddicients sum to 1, we also provide plots of the coefficients mapped to the simplex.

![Const noise](play/viz-interpolators/ConstantNoise.png?raw=true)
![EDS](play/viz-interpolators/EncodingDecodingStochastic.png?raw=true)
![LD](play/viz-interpolators/LinearDeterministic.png?raw=true)
![LS](play/viz-interpolators/LinearStochastic.png?raw=true)
![SD](play/viz-interpolators/SquaredDeterministic.png?raw=true)
![SS](play/viz-interpolators/SquaredStochastic.png?raw=true)
![TS](play/viz-interpolators/TrigonometricStochastic.png?raw=true)


## Exact implementation

The exact implementation of stochastic interpolants for Gaussian distributions and Gaussian mixtures is provided in `sinterp.exact_si_gaussian` and `sinterp.exact_si_gmm` respectively. The equations are taken from page 35 [Stochastic interpolants](https://arxiv.org/abs/2209.03003).

For example, here is an example that maps from a 3 mode 1D GMM to a 2 mode 1D GMM. 

![drift of LD for a GMM 3-2](play/pbs-fields/b_LinearStochastic-3-2.png?raw=true)
![velocity of LD for a GMM 3-2](play/pbs-fields/v_LinearStochastic-3-2.png?raw=true)
![score of LD for a GMM 3-2](play/pbs-fields/s_LinearStochastic-3-2.png?raw=true)
![trajectories of LD for a GMM 3-2](play/pbs-fields/trajectories_LinearStochastic-3-2.png?raw=true)

You can find a few other examples in the `play/pbs-fields` directory.
The exact GMM calcs support any dimensionality, and any number of gaussian distributions.

<!-- ## Losses

$$
\mathcal L
$$ -->
