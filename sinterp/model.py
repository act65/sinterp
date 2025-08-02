import jax
import jax.numpy as jnp
from jax import jacfwd
from abc import ABC, abstractmethod
from functools import partial

from sinterp.interpolants import Interpolator

class BaseGenerativeModel(ABC):
    """
    Abstract Base Class defining the public interface for all generative models.
    
    This ensures that any model, regardless of how it's constructed (one-sided, 
    two-sided, etc.), can provide the fundamental quantities of the framework.
    """
    def __init__(self, interpolator: Interpolator):
        self.interpolator = interpolator

    @abstractmethod
    def get_velocity_b(self, params, x_t, t):
        """Returns the full probability flow velocity b(t,x). Used for ODE sampling."""
        pass

    @abstractmethod
    def get_velocity_v(self, params, x_t, t):
        """Returns the deterministic part of the velocity v(t,x). A learned component."""
        pass

    @abstractmethod
    def get_score_s(self, params, x_t, t):
        """Returns the score s(t,x). Used for SDE sampling."""
        pass

    @abstractmethod
    def get_denoiser_eta0(self, params, x_t, t):
        """Returns the denoiser for the initial data, E[x₀|xₜ]."""
        pass

    @abstractmethod
    def get_denoiser_eta1(self, params, x_t, t):
        """Returns the denoiser for the final data, E[x₁|xₜ]."""
        pass

    @abstractmethod
    def get_denoiser_etaz(self, params, x_t, t):
        """Returns the denoiser for the latent noise, E[z|xₜ]."""
        pass


class TwoSidedModel(BaseGenerativeModel):
    """
    A two-sided generative model constructed from two learned components:
    1. v_model: A network for the deterministic velocity v(t,x) = α'(t)η₀ + β'(t)η₁
    2. eta_z_model: A network for the noise denoiser E[z|xₜ]
    
    This is the recommended, well-posed approach for two-sided flows.
    """
    def __init__(self, interpolator: Interpolator, v_model, eta_z_model):
        super().__init__(interpolator)
        self.v_model = v_model
        self.eta_z_model = eta_z_model

    @partial(jax.jit, static_argnums=(0,))
    def _solve_for_data_denoisers(self, v_val, eta_z_val, x_t, t):
        """
        Solves the 2x2 linear system for η₀ and η₁ given v and ηz.
        Eq 1: v = α'η₀ + β'η₁
        Eq 2: xₜ - γηz = αη₀ + βη₁
        """
        alpha_t, beta_t, gamma_t = self.interpolator.alpha(t), self.interpolator.beta(t), self.interpolator.gamma(t)
        alpha_dot_t, beta_dot_t = self.interpolator.dalphadt(t), self.interpolator.dbetadt(t)

        # Determinant of the system matrix A = [[α', β'], [α, β]]
        det = alpha_dot_t * beta_t - beta_dot_t * alpha_t
        safe_det = jnp.where(jnp.abs(det) > 1e-9, det, 1.0)

        # RHS vector B = [v, x_t - γη_z]
        rhs2 = x_t - gamma_t * eta_z_val

        # Solve using Cramer's rule / inverse matrix
        eta_0 = (beta_t * v_val - beta_dot_t * rhs2) / safe_det
        eta_1 = (alpha_dot_t * rhs2 - alpha_t * v_val) / safe_det
        
        is_defined = jnp.abs(det) > 1e-9
        return jnp.where(is_defined, eta_0, 0.0), jnp.where(is_defined, eta_1, 0.0)

    def get_velocity_v(self, params, x_t, t):
        v_params, _ = params
        return self.v_model(v_params, x_t, t)

    def get_denoiser_etaz(self, params, x_t, t):
        _, eta_z_params = params
        return self.eta_z_model(eta_z_params, x_t, t)

    def get_velocity_b(self, params, x_t, t):
        v_val = self.get_velocity_v(params, x_t, t)
        eta_z_val = self.get_denoiser_etaz(params, x_t, t)
        gamma_dot_t = self.interpolator.dgammadt(t)
        return v_val + gamma_dot_t * eta_z_val

    def get_score_s(self, params, x_t, t):
        eta_z_val = self.get_denoiser_etaz(params, x_t, t)
        gamma_t = self.interpolator.gamma(t)
        safe_gamma_t = jnp.where(jnp.abs(gamma_t) > 1e-9, gamma_t, 1.0)
        score = -eta_z_val / safe_gamma_t
        return jnp.where(jnp.abs(gamma_t) > 1e-9, score, jnp.zeros_like(score))

    def get_denoiser_eta0(self, params, x_t, t):
        v_val = self.get_velocity_v(params, x_t, t)
        eta_z_val = self.get_denoiser_etaz(params, x_t, t)
        eta_0, _ = self._solve_for_data_denoisers(v_val, eta_z_val, x_t, t)
        return eta_0

    def get_denoiser_eta1(self, params, x_t, t):
        v_val = self.get_velocity_v(params, x_t, t)
        eta_z_val = self.get_denoiser_etaz(params, x_t, t)
        _, eta_1 = self._solve_for_data_denoisers(v_val, eta_z_val, x_t, t)
        return eta_1


class OneSidedModel(BaseGenerativeModel):
    """
    A one-sided generative model constructed from a single learned component:
    1. velocity_b_model: A network for the full probability flow velocity b(t,x).
    
    This is well-posed because for a one-sided flow, we only have two unknown
    denoisers (ηz and η₁), which can be solved for algebraically from b(t,x).
    """
    def __init__(self, interpolator: Interpolator, velocity_b_model):
        super().__init__(interpolator)
        self.velocity_b_model = velocity_b_model

    @partial(jax.jit, static_argnums=(0,))
    def _solve_for_denoisers(self, b_val, x_t, t):
        """
        Solves the 2x2 linear system for ηz and η₁ given b.
        Here, ηz is the denoiser for the initial noise z (i.e., it plays the role of η₀).
        Eq 1: b = α'ηz + β'η₁
        Eq 2: xₜ = αηz + βη₁
        """
        alpha_t, beta_t = self.interpolator.alpha(t), self.interpolator.beta(t)
        alpha_dot_t, beta_dot_t = self.interpolator.dalphadt(t), self.interpolator.dbetadt(t)

        # Determinant of the system matrix A = [[α', β'], [α, β]]
        det = alpha_dot_t * beta_t - beta_dot_t * alpha_t
        safe_det = jnp.where(jnp.abs(det) > 1e-9, det, 1.0)

        # Solve using Cramer's rule / inverse matrix
        eta_z = (beta_t * b_val - beta_dot_t * x_t) / safe_det
        eta_1 = (alpha_dot_t * x_t - alpha_t * b_val) / safe_det
        
        is_defined = jnp.abs(det) > 1e-9
        return jnp.where(is_defined, eta_z, 0.0), jnp.where(is_defined, eta_1, 0.0)

    def get_velocity_b(self, params, x_t, t):
        return self.velocity_b_model(params, x_t, t)

    def get_denoiser_eta0(self, params, x_t, t):
        # In the one-sided case, x₀ is the noise z, so η₀ is ηz.
        b_val = self.get_velocity_b(params, x_t, t)
        eta_z, _ = self._solve_for_denoisers(b_val, x_t, t)
        return eta_z

    def get_denoiser_eta1(self, params, x_t, t):
        b_val = self.get_velocity_b(params, x_t, t)
        _, eta_1 = self._solve_for_denoisers(b_val, x_t, t)
        return eta_1

    def get_denoiser_etaz(self, params, x_t, t):
        # In the one-sided formulation, there is no separate latent z.
        # We return 0 by convention. The relevant noise denoiser is eta_0.
        return jnp.zeros_like(x_t)

    def get_score_s(self, params, x_t, t):
        # The score is with respect to the initial noise z, which is x₀.
        # s(t,x) = -η₀(t,x) / α(t)
        eta_0_val = self.get_denoiser_eta0(params, x_t, t)
        alpha_t = self.interpolator.alpha(t)
        safe_alpha_t = jnp.where(jnp.abs(alpha_t) > 1e-9, alpha_t, 1.0)
        score = -eta_0_val / safe_alpha_t
        return jnp.where(jnp.abs(alpha_t) > 1e-9, score, jnp.zeros_like(score))

    def get_velocity_v(self, params, x_t, t):
        # For a one-sided flow, v(t,x) is the part of the velocity NOT associated
        # with the initial noise. This leaves only the data component.
        # v(t,x) = β'(t)η₁(t,x)
        eta_1_val = self.get_denoiser_eta1(params, x_t, t)
        beta_dot_t = self.interpolator.dbetadt(t)
        return beta_dot_t * eta_1_val