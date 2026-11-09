The repository provides implementations of [Stochastic interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797).

Specifically, it provides implementations of;

- the many variants of interpolant
- couplings between the target and source distributions
- various different losses (for the drift, velocity, and score fields)
- stochastic fields (the core math of stochastic interpolants relating denoised values to vector fields)
- an exact implementation of stochastic interpolants for Gaussian distributions and Gaussian mixtures

## Stochastic fields

The class `StochasticField` provides all the equations for the vector fields and expectations. Our interpolant defines;

$$
x_t = \alpha(t)x_0 + \beta(t)x_1 + \gamma(t)z
$$

Here, $x_0$ is a sample from $p_0$, $x_1$ is a sample from $p_1$, and $z$ is a random noise sample. The functions α(t), β(t), and γ(t) control the interpolation schedule.

From the paper ["Stochastic Interpolants"](https://arxiv.org/abs/2303.08797), eqn 4.4 and the related section shows how the fundamental quantities of the generative process—the **velocity `b(t,x)`** and the **score `s(t,x)`**—can be broken down and expressed in terms of three "denoisers":

*   $\eta_\theta(t,x) = E[x_0\mid x_t]$: The expected value of the initial sample `x_1` given the interpolant's state `x_t` at time `t`.
*   $\eta_1(t,x) = E[x_1\mid x_t]$: The expected value of the final sample `x₁` given `xₜ`.
*   $\eta_z(t,x) = E[z\mid x_t]$: The expected value of the latent noise `z` given `xₜ`.

The vector fields:

$$
\begin{align}
b(t,x) &= \alpha'(t)\eta_0(t,x) + \beta'(t)\eta_1(t,x) + \gamma'(t)\eta_z(t,x) \tag{the flow ODE velocity fn}\\
s(t,x) &= -\gamma (t)^{-1} \eta_z(t,x) \tag{the score fn} \\
v(t,x) &= \alpha'(t)\eta_0(t,x) + \beta'(t)\eta_1(t,x) \tag{the deterministic velocity, aka the drift}
\end{align}
$$

## Couplings

We implement the;

- independent coupling. `p(x0, x1) = p(x1)p(x0)`
- conditional coupling. `p(x0, x1) = p(x1 | x0)p(x0)`
- minibatch optimal transport coupling (see [Multisample Flow Matching](http://proceedings.mlr.press/v202/pooladian23a/pooladian23a.pdf) and [Stochastic interpolants with data-dependent couplings](https://arxiv.org/abs/2310.03725) for more details)

## Interpolants

We plot the interpolants defined in `sinterp/interpolants.py`.
Because many of the interpolants require that the coeddicients sum to 1, we also provide plots of the coefficients mapped to the simplex.

![Const noise](../images/viz-interpolators/ConstantNoise.png?raw=true)
![EDS](../images/viz-interpolators/EncodingDecodingStochastic.png?raw=true)
![LD](../images/viz-interpolators/LinearDeterministic.png?raw=true)
![LS](../images/viz-interpolators/LinearStochastic.png?raw=true)
![SD](../images/viz-interpolators/SquaredDeterministic.png?raw=true)
![SS](../images/viz-interpolators/SquaredStochastic.png?raw=true)
![TS](../images/viz-interpolators/TrigonometricStochastic.png?raw=true)


## Exact implementation

The exact implementation of stochastic interpolants for Gaussian distributions and Gaussian mixtures is provided in `sinterp.exact_si_gaussian` and `sinterp.exact_si_gmm` respectively. The equations are taken from page 35 [Stochastic interpolants](https://arxiv.org/abs/2209.03003).

For example, here is an example that maps from a 3 mode 1D GMM to a 2 mode 1D GMM. 

![drift of LD for a GMM 3-2](../images/pbs-fields/b_LinearStochastic-3-2.png?raw=true)
![velocity of LD for a GMM 3-2](../images/pbs-fields/v_LinearStochastic-3-2.png?raw=true)
![score of LD for a GMM 3-2](../images/pbs-fields/s_LinearStochastic-3-2.png?raw=true)
![trajectories of LD for a GMM 3-2](../images/pbs-fields/trajectories_LinearStochastic-3-2.png?raw=true)

You can find a few other examples in the `images/pbs-fields` directory.
The exact GMM calcs support any dimensionality, and any number of gaussian distributions.