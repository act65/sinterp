import jax
import jax.numpy as jnp
import equinox as eqx

from sinterp.interpolants import Interpolator

class StochasticField(eqx.Module):
    """
    Base Class defining the public interface for a stochastic field.
    """
    # We mark interpolator as static assuming it contains just math logic/config 
    # and no trainable parameters.
    interpolator: Interpolator = eqx.field(static=True)

    def __init__(self, interpolator: Interpolator):
        self.interpolator = interpolator

    # --- Interface (No params arg needed!) ---

    def get_velocity_b(self, x_t, t, *args):
        """Returns the full probability flow velocity b(t,x)."""
        raise NotImplementedError

    def get_velocity_v(self, x_t, t, *args):
        """Returns the deterministic part of the velocity v(t,x)."""
        raise NotImplementedError

    def get_score_s(self, x_t, t, *args):
        """Returns the score s(t,x)."""
        raise NotImplementedError

    def get_denoiser_eta0(self, x_t, t, *args):
        """Returns the denoiser for the initial data, E[x₀|xₜ]."""
        raise NotImplementedError

    def get_denoiser_eta1(self, x_t, t, *args):
        """Returns the denoiser for the final data, E[x₁|xₜ]."""
        raise NotImplementedError

    def get_denoiser_etaz(self, x_t, t, *args):
        """Returns the denoiser for the latent noise, E[z|xₜ]."""
        raise NotImplementedError

    # --- Losses (Params are implicit in 'self') ---

    def loss_b(self, x, y, e, t, *args):
        """Flow matching loss: match the full probability flow velocity b."""
        i = self.interpolator(x, y, e, t)
        di = self.interpolator.didt(x, y, e, t)
        b_val = self.get_velocity_b(i, t, *args)
        
        def _compute_loss(_b, _di):
            return jnp.mean((_b - _di)**2)
        
        losses = jax.tree_util.tree_map(_compute_loss, b_val, di)
        return jax.tree_util.tree_reduce(lambda x, y: x + y, losses)

    def loss_v(self, x, y, e, t, *args):
        """Deterministic velocity matching loss."""
        i = self.interpolator(x, y, e, t)
        di = self.interpolator.didt_deterministic(x, y, t)
        v_val = self.get_velocity_v(i, t, *args)
        
        def _compute_loss(_v, _di):
            return jnp.mean((_v - _di)**2)
        
        losses = jax.tree_util.tree_map(_compute_loss, v_val, di)
        return jax.tree_util.tree_reduce(lambda x, y: x + y, losses)

    def loss_s(self, x, y, e, t, *args):
        """Score matching loss."""
        i = self.interpolator.deterministic(x, y, t)
        st = self.get_score_s(i, t, *args)
        gamma = self.interpolator.gamma(t)
        
        safe_gamma = jnp.where(jnp.abs(gamma) > 1e-5, gamma, 1.0)
        
        def _compute_loss(_st, _x):
            return 0.5 * jnp.linalg.norm(_st)**2 + (1/safe_gamma) * ((_st * _x).sum())
            
        losses = jax.tree_util.tree_map(_compute_loss, st, x)
        loss = jax.tree_util.tree_reduce(lambda x, y: x + y, losses)
        
        return jnp.where(jnp.abs(gamma) > 1e-5, loss, 0.0)

    def loss_eta0(self, x, y, e, t, *args):
        i = self.interpolator(x, y, e, t)
        eta0 = self.get_denoiser_eta0(i, t, *args)
        
        def _compute_loss(_e0, _x):
            return jnp.mean((_e0 - _x)**2)
            
        losses = jax.tree_util.tree_map(_compute_loss, eta0, x)
        return jax.tree_util.tree_reduce(lambda x, y: x + y, losses)

    def loss_eta1(self, x, y, e, t, *args):
        i = self.interpolator(x, y, e, t)
        eta1 = self.get_denoiser_eta1(i, t, *args)
        
        def _compute_loss(_e1, _y):
            return jnp.mean((_e1 - _y)**2)
            
        losses = jax.tree_util.tree_map(_compute_loss, eta1, y)
        return jax.tree_util.tree_reduce(lambda x, y: x + y, losses)

    def loss_etaz(self, x, y, e, t, *args):
        i = self.interpolator(x, y, e, t)
        etaz = self.get_denoiser_etaz(i, t, *args)
        
        def _compute_loss(_ez, _e):
            return jnp.mean((_ez - _e)**2)
            
        losses = jax.tree_util.tree_map(_compute_loss, etaz, e)
        return jax.tree_util.tree_reduce(lambda x, y: x + y, losses)


class TwoSidedField(StochasticField):
    """Base class for all Two-Sided Stochastic Interpolants."""

    def _solve_for_data_denoisers(self, v_val, eta_z_val, x_t, t):
        """
        Solves the 2x2 linear system.
        Note: Removed explicit @jit. When you jit the total_loss externally,
        JAX will trace through this efficiently.
        """
        alpha_t = self.interpolator.alpha(t)
        beta_t = self.interpolator.beta(t)
        gamma_t = self.interpolator.gamma(t)
        
        alpha_dot_t = self.interpolator.dalphadt(t)
        beta_dot_t = self.interpolator.dbetadt(t)

        det = alpha_dot_t * beta_t - beta_dot_t * alpha_t
        safe_det = jnp.where(jnp.abs(det) > 1e-9, det, 1.0)
        is_defined = jnp.abs(det) > 1e-9

        def _solve(_v, _ez, _xt):
            rhs_geom = _xt - gamma_t * _ez
            eta_0 = (beta_t * _v - beta_dot_t * rhs_geom) / safe_det
            eta_1 = (alpha_dot_t * rhs_geom - alpha_t * _v) / safe_det
            return jnp.where(is_defined, eta_0, 0.0), jnp.where(is_defined, eta_1, 0.0)

        out = jax.tree_util.tree_map(_solve, v_val, eta_z_val, x_t)
        
        # When _solve returns (eta_0, eta_1), tree_map preserves the outer structure.
        # If inputs were dicts, out is a dict of tuples.
        # We need to transpose this to get a tuple of dicts.
        
        outer_structure = jax.tree_util.tree_structure(x_t)
        inner_structure = jax.tree_util.tree_structure((0,0))
        return jax.tree_util.tree_transpose(outer_structure, inner_structure, out)

    def get_denoiser_etaz(self, x_t, t, *args):
        eta0 = self.get_denoiser_eta0(x_t, t, *args)
        eta1 = self.get_denoiser_eta1(x_t, t, *args)
        
        alpha = self.interpolator.alpha(t)
        beta = self.interpolator.beta(t)
        gamma = self.interpolator.gamma(t)
        
        def _compute_etaz(_xt, _e0, _e1):
            numerator = _xt - alpha * _e0 - beta * _e1
            safe_gamma = jnp.where(jnp.abs(gamma) > 1e-5, gamma, 1.0)
            eta_z = numerator / safe_gamma
            return jnp.where(jnp.abs(gamma) > 1e-5, eta_z, jnp.zeros_like(_xt))

        return jax.tree_util.tree_map(_compute_etaz, x_t, eta0, eta1)


class TwoSidedField_v_etaz(TwoSidedField):
    # These contain the trainable weights
    model_v: eqx.Module
    model_etaz: eqx.Module

    def __init__(self, interpolator, model_v, model_etaz):
        super().__init__(interpolator)
        self.model_v = model_v
        self.model_etaz = model_etaz

    def get_velocity_v(self, x_t, t, *args):
        return self.model_v(x_t, t, *args)

    def get_denoiser_etaz(self, x_t, t, *args):
        return self.model_etaz(x_t, t, *args)

    def get_velocity_b(self, x_t, t, *args):
        v = self.get_velocity_v(x_t, t, *args)
        eta_z = self.get_denoiser_etaz(x_t, t, *args)
        gamma_dot = self.interpolator.dgammadt(t)
        return jax.tree_util.tree_map(lambda _v, _ez: _v + gamma_dot * _ez, v, eta_z)

    def get_score_s(self, x_t, t, *args):
        eta_z = self.get_denoiser_etaz(x_t, t, *args)
        gamma = self.interpolator.gamma(t)
        safe_gamma = jnp.where(jnp.abs(gamma) > 1e-5, gamma, 1.0)
        
        def _compute_score(_ez):
            score = -_ez / safe_gamma
            return jnp.where(jnp.abs(gamma) > 1e-5, score, jnp.zeros_like(_ez))
            
        return jax.tree_util.tree_map(_compute_score, eta_z)

    def get_denoiser_eta0(self, x_t, t, *args):
        v = self.get_velocity_v(x_t, t, *args)
        eta_z = self.get_denoiser_etaz(x_t, t, *args)
        eta_0, _ = self._solve_for_data_denoisers(v, eta_z, x_t, t)
        return eta_0

    def get_denoiser_eta1(self, x_t, t, *args):
        v = self.get_velocity_v(x_t, t, *args)
        eta_z = self.get_denoiser_etaz(x_t, t, *args)
        _, eta_1 = self._solve_for_data_denoisers(v, eta_z, x_t, t)
        return eta_1

    def total_loss(self, x, y, e, t, *args):
        return self.loss_v(x, y, e, t, *args) + self.loss_etaz(x, y, e, t, *args)


class TwoSidedField_b_s(TwoSidedField):
    model_b: eqx.Module
    model_s: eqx.Module

    def __init__(self, interpolator, model_b, model_s):
        super().__init__(interpolator)
        self.model_b = model_b
        self.model_s = model_s

    def get_velocity_b(self, x_t, t, *args):
        return self.model_b(x_t, t, *args)

    def get_score_s(self, x_t, t, *args):
        return self.model_s(x_t, t, *args)

    def get_denoiser_etaz(self, x_t, t, *args):
        s = self.get_score_s(x_t, t, *args)
        return -self.interpolator.gamma(t) * s

    def get_velocity_v(self, x_t, t, *args):
        b = self.get_velocity_b(x_t, t, *args)
        s = self.get_score_s(x_t, t, *args)
        gamma = self.interpolator.gamma(t)
        gamma_dot = self.interpolator.dgammadt(t)
        return b + (gamma * gamma_dot) * s

    def get_denoiser_eta0(self, x_t, t, *args):
        v = self.get_velocity_v(x_t, t, *args)
        eta_z = self.get_denoiser_etaz(x_t, t, *args)
        eta_0, _ = self._solve_for_data_denoisers(v, eta_z, x_t, t)
        return eta_0

    def get_denoiser_eta1(self, x_t, t, *args):
        v = self.get_velocity_v(x_t, t, *args)
        eta_z = self.get_denoiser_etaz(x_t, t, *args)
        _, eta_1 = self._solve_for_data_denoisers(v, eta_z, x_t, t)
        return eta_1

    def total_loss(self, x, y, e, t, *args):
        return self.loss_b(x, y, e, t, *args) + self.loss_s(x, y, e, t, *args)


class TwoSidedField_x0_x1(TwoSidedField):
    model_eta0: eqx.Module
    model_eta1: eqx.Module

    def __init__(self, interpolator, model_eta0, model_eta1):
        super().__init__(interpolator)
        self.model_eta0 = model_eta0
        self.model_eta1 = model_eta1

    def get_denoiser_eta0(self, x_t, t, *args):
        return self.model_eta0(x_t, t, *args)

    def get_denoiser_eta1(self, x_t, t, *args):
        return self.model_eta1(x_t, t, *args)

    def get_velocity_v(self, x_t, t, *args):
        eta0 = self.get_denoiser_eta0(x_t, t, *args)
        eta1 = self.get_denoiser_eta1(x_t, t, *args)
        return (self.interpolator.dalphadt(t) * eta0 + 
                self.interpolator.dbetadt(t) * eta1)

    def get_denoiser_etaz(self, x_t, t, *args):
        eta0 = self.get_denoiser_eta0(x_t, t, *args)
        eta1 = self.get_denoiser_eta1(x_t, t, *args)
        
        alpha = self.interpolator.alpha(t)
        beta = self.interpolator.beta(t)
        gamma = self.interpolator.gamma(t)
        
        numerator = x_t - alpha * eta0 - beta * eta1
        
        safe_gamma = jnp.where(jnp.abs(gamma) > 1e-5, gamma, 1.0)
        eta_z = numerator / safe_gamma
        return jnp.where(jnp.abs(gamma) > 1e-5, eta_z, jnp.zeros_like(x_t))

    def get_velocity_b(self, x_t, t, *args):
        v = self.get_velocity_v(x_t, t, *args)
        eta_z = self.get_denoiser_etaz(x_t, t, *args)
        return v + self.interpolator.dgammadt(t) * eta_z

    def get_score_s(self, x_t, t, *args):
        eta_z = self.get_denoiser_etaz(x_t, t, *args)
        gamma = self.interpolator.gamma(t)
        safe_gamma = jnp.where(jnp.abs(gamma) > 1e-5, gamma, 1.0)
        score = -eta_z / safe_gamma
        return jnp.where(jnp.abs(gamma) > 1e-5, score, jnp.zeros_like(score))

    def total_loss(self, x, y, e, t, *args):
        return self.loss_eta0(x, y, e, t, *args) + self.loss_eta1(x, y, e, t, *args)


class TwoSidedField_b_eta1(TwoSidedField):
    model_b: eqx.Module
    model_eta1: eqx.Module

    def __init__(self, interpolator, model_b, model_eta1):
        super().__init__(interpolator)
        self.model_b = model_b
        self.model_eta1 = model_eta1

    def get_velocity_b(self, x_t, t, *args):
        return self.model_b(x_t, t, *args)

    def get_denoiser_eta1(self, x_t, t, *args):
        return self.model_eta1(x_t, t, *args)

    def _solve_for_eta0_etaz(self, b_val, eta1_val, x_t, t):
        alpha = self.interpolator.alpha(t)
        beta = self.interpolator.beta(t)
        gamma = self.interpolator.gamma(t)
        
        alpha_dot = self.interpolator.dalphadt(t)
        beta_dot = self.interpolator.dbetadt(t)
        gamma_dot = self.interpolator.dgammadt(t)
        
        det = alpha * gamma_dot - alpha_dot * gamma
        safe_det = jnp.where(jnp.abs(det) > 1e-9, det, 1.0)
        is_defined = jnp.abs(det) > 1e-9
        
        def _solve(_b, _e1, _xt):
            rhs_pos = _xt - beta * _e1
            rhs_vel = _b - beta_dot * _e1
            
            eta0 = (rhs_pos * gamma_dot - rhs_vel * gamma) / safe_det
            etaz = (alpha * rhs_vel - alpha_dot * rhs_pos) / safe_det
            
            return jnp.where(is_defined, eta0, 0.0), jnp.where(is_defined, etaz, 0.0)

        out = jax.tree_util.tree_map(_solve, b_val, eta1_val, x_t)
        outer_structure = jax.tree_util.tree_structure(x_t)
        inner_structure = jax.tree_util.tree_structure((0,0))
        return jax.tree_util.tree_transpose(outer_structure, inner_structure, out)

    def get_denoiser_eta0(self, x_t, t, *args):
        b = self.get_velocity_b(x_t, t, *args)
        eta1 = self.get_denoiser_eta1(x_t, t, *args)
        eta0, _ = self._solve_for_eta0_etaz(b, eta1, x_t, t)
        return eta0

    def get_denoiser_etaz(self, x_t, t, *args):
        b = self.get_velocity_b(x_t, t, *args)
        eta1 = self.get_denoiser_eta1(x_t, t, *args)
        _, etaz = self._solve_for_eta0_etaz(b, eta1, x_t, t)
        return etaz

    def get_velocity_v(self, x_t, t, *args):
        eta0 = self.get_denoiser_eta0(x_t, t, *args)
        eta1 = self.get_denoiser_eta1(x_t, t, *args)
        
        alpha_dot = self.interpolator.dalphadt(t)
        beta_dot = self.interpolator.dbetadt(t)
        
        return jax.tree_util.tree_map(lambda e0, e1: alpha_dot * e0 + beta_dot * e1, eta0, eta1)

    def get_score_s(self, x_t, t, *args):
        etaz = self.get_denoiser_etaz(x_t, t, *args)
        gamma = self.interpolator.gamma(t)
        safe_gamma = jnp.where(jnp.abs(gamma) > 1e-5, gamma, 1.0)
        
        def _compute_score(_ez):
            score = -_ez / safe_gamma
            return jnp.where(jnp.abs(gamma) > 1e-5, score, jnp.zeros_like(_ez))
            
        return jax.tree_util.tree_map(_compute_score, etaz)

    def total_loss(self, x, y, e, t, *args):
        return self.loss_b(x, y, e, t, *args) + self.loss_eta1(x, y, e, t, *args)


class OneSidedFlow(StochasticField):
    def get_denoiser_eta0(self, x_t, t, *args):
        return jax.tree_util.tree_map(jnp.zeros_like, x_t)


class OneSidedField_b(StochasticField):
    """
    Solves for denoisers assuming interpolant x_t = gamma*e + beta*x1 (alpha=0).
    Here eta_z estimates e, eta_1 estimates x1. eta_0 is 0.
    """
    net: eqx.Module

    def __init__(self, interpolator, net):
        super().__init__(interpolator)
        self.net = net

    def _solve_for_denoisers(self, b_val, x_t, t):
        # We solve:
        # 1. x_t = gamma * eta_z + beta * eta_1
        # 2. b   = gamma' * eta_z + beta' * eta_1
        gamma_t = self.interpolator.gamma(t)
        beta_t = self.interpolator.beta(t)
        gamma_dot_t = self.interpolator.dgammadt(t)
        beta_dot_t = self.interpolator.dbetadt(t)

        det = gamma_dot_t * beta_t - beta_dot_t * gamma_t
        safe_det = jnp.where(jnp.abs(det) > 1e-9, det, 1.0)
        is_defined = jnp.abs(det) > 1e-9

        def _solve(_bv, _xt):
            eta_z = (beta_t * _bv - beta_dot_t * _xt) / safe_det
            eta_1 = (gamma_dot_t * _xt - gamma_t * _bv) / safe_det
            return jnp.where(is_defined, eta_z, 0.0), jnp.where(is_defined, eta_1, 0.0)

        out = jax.tree_util.tree_map(_solve, b_val, x_t)
        
        # See note in TwoSidedField._solve_for_data_denoisers
        outer_structure = jax.tree_util.tree_structure(x_t)
        inner_structure = jax.tree_util.tree_structure((0,0))
        return jax.tree_util.tree_transpose(outer_structure, inner_structure, out)

    def get_velocity_b(self, x_t, t, *args):
        return self.net(x_t, t, *args)

    def get_denoiser_eta0(self, x_t, t, *args):
        # In this formulation (alpha=0), eta0 is 0.
        return jax.tree_util.tree_map(jnp.zeros_like, x_t)

    def get_denoiser_etaz(self, x_t, t, *args):
        b_val = self.get_velocity_b(x_t, t, *args)
        eta_z, _ = self._solve_for_denoisers(b_val, x_t, t)
        return eta_z

    def get_denoiser_eta1(self, x_t, t, *args):
        b_val = self.get_velocity_b(x_t, t, *args)
        _, eta_1 = self._solve_for_denoisers(b_val, x_t, t)
        return eta_1

    def get_score_s(self, x_t, t, *args):
        # s = -eta_z / gamma
        eta_z_val = self.get_denoiser_etaz(x_t, t, *args)
        gamma_t = self.interpolator.gamma(t)
        safe_gamma = jnp.where(jnp.abs(gamma_t) > 1e-9, gamma_t, 1.0)
        
        def _compute_score(_ez):
            score = -_ez / safe_gamma
            return jnp.where(jnp.abs(gamma_t) > 1e-9, score, jnp.zeros_like(_ez))
            
        return jax.tree_util.tree_map(_compute_score, eta_z_val)

    def get_velocity_v(self, x_t, t, *args):
        # v = beta' * eta_1 (since alpha=0)
        eta_1_val = self.get_denoiser_eta1(x_t, t, *args)
        beta_dot_t = self.interpolator.dbetadt(t)
        return jax.tree_util.tree_map(lambda _e1: beta_dot_t * _e1, eta_1_val)

    def total_loss(self, x, y, e, t, *args):
        return self.loss_b(x, y, e, t, *args)


class OneSidedField_eta1(StochasticField):
    """
    One-sided field learning eta_1 directly.
    """
    net: eqx.Module

    def __init__(self, interpolator, net):
        super().__init__(interpolator)
        self.net = net
            
    def _solve_for_etaz(self, eta_1_val, x_t, t):
        # x_t = gamma * eta_z + beta * eta_1
        gamma_t = self.interpolator.gamma(t)
        beta_t = self.interpolator.beta(t)
        
        # FIX: Increased epsilon from 1e-9 to 1e-4 to prevent explosion near t=1
        safe_gamma = jnp.where(jnp.abs(gamma_t) > 1e-4, gamma_t, 1.0)
        
        def _compute_ez(_e1, _xt):
            eta_z = (_xt - beta_t * _e1) / safe_gamma
            # FIX: Ensure we mask out the result where gamma is too small
            # (Velocity calculation should handle the singularity, but this helps)
            return jnp.where(jnp.abs(gamma_t) > 1e-4, eta_z, jnp.zeros_like(_xt))
            
        return jax.tree_util.tree_map(_compute_ez, eta_1_val, x_t)

    def get_denoiser_eta1(self, x_t, t, *args):
        return self.net(x_t, t, *args)

    def get_denoiser_eta0(self, x_t, t, *args):
        return jax.tree_util.tree_map(jnp.zeros_like, x_t)

    def get_denoiser_etaz(self, x_t, t, *args):
        eta_1_val = self.get_denoiser_eta1(x_t, t, *args)
        return self._solve_for_etaz(eta_1_val, x_t, t)

    def get_velocity_b(self, x_t, t, *args):
        # We model the velocity as pushing x_t towards the predicted x_1.
        # For OneSidedLinear: v = (x_1 - x_t) / (1 - t)
        
        eta_1_val = self.get_denoiser_eta1(x_t, t, *args)
        
        # --- ROBUSTNESS FIX ---
        # The integrator evaluates exactly at t=1.0. 
        # We soft-clamp t to 0.99 to limit velocity magnification.
        t_safe = jnp.minimum(t, 0.99)
        
        # Recompute coefficients with safe time        
        gamma_t = self.interpolator.gamma(t_safe)
        beta_t = self.interpolator.beta(t_safe)
        gamma_dot = self.interpolator.dgammadt(t_safe)
        beta_dot = self.interpolator.dbetadt(t_safe)
        
        # safe_gamma must not be near zero. 
        # With t_safe <= 0.99, gamma >= 0.01 (for Linear).
        safe_gamma = jnp.maximum(gamma_t, 1e-3)

        # Solve for eta_z using the safe time
        # BUG??? this is specific for linear interp?
        eta_z_val = jax.tree_util.tree_map(
            lambda _e1, _xt: (_xt - beta_t * _e1) / safe_gamma, 
            eta_1_val, x_t
        )

        # Compute velocity
        return jax.tree_util.tree_map(
            lambda _ez, _e1: gamma_dot * _ez + beta_dot * _e1, 
            eta_z_val, eta_1_val
        )

    def get_velocity_v(self, x_t, t, *args):
        eta_1_val = self.get_denoiser_eta1(x_t, t, *args)
        beta_dot = self.interpolator.dbetadt(t)
        return jax.tree_util.tree_map(lambda _e1: beta_dot * _e1, eta_1_val)

    def get_score_s(self, x_t, t, *args):
        # s = -eta_z / gamma
        eta_z_val = self.get_denoiser_etaz(x_t, t, *args)
        gamma_t = self.interpolator.gamma(t)
        safe_gamma = jnp.where(jnp.abs(gamma_t) > 1e-9, gamma_t, 1.0)
        
        def _compute_score(_ez):
            score = -_ez / safe_gamma
            return jnp.where(jnp.abs(gamma_t) > 1e-9, score, jnp.zeros_like(_ez))
            
        return jax.tree_util.tree_map(_compute_score, eta_z_val)

    def total_loss(self, x, y, e, t, *args):
        return self.loss_eta1(x, y, e, t, *args)
