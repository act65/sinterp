"""Core stochastic-interpolant mathematics.

Provides interpolant schedules (:class:`Interpolator` and subclasses), couplings
between source and target distributions, and stochastic fields
(:class:`StochasticField`) that relate denoisers to the velocity/score/drift
vector fields. See ``sinterp/README.md`` for the equations and conventions.
"""

from sinterp.interpolants import (
    Interpolator,
    LinearDeterministic,
    LinearStochastic,
    SquaredDeterministic,
    SquaredStochastic,
    TrigonometricStochastic,
    LinearVPStochastic,
    TrigonometricVPStochastic,
    EncodingDecodingStochastic,
    LowNoiseEncodingDecodingStochastic,
    ConstantNoise,
    OneSidedLinear,
    OneSidedSBDM,
)
from sinterp.couplings import (
    Coupling,
    IndependentCoupling,
    SinkhornCoupling,
    ExactOTCoupling,
    ExactConditionalCoupling,
    C2OTCoupling,
)
from sinterp.stochasticfield import (
    StochasticField,
    TwoSidedField,
    TwoSidedField_v_etaz,
    TwoSidedField_b_s,
    TwoSidedField_x0_x1,
    TwoSidedField_b_eta1,
    OneSidedFlow,
    OneSidedField_b,
    OneSidedField_eta1,
)
from sinterp.field_factory import field_factory, make_default_net_constructor

__version__ = "0.3.0"

__all__ = [
    "Interpolator",
    "LinearDeterministic",
    "LinearStochastic",
    "SquaredDeterministic",
    "SquaredStochastic",
    "TrigonometricStochastic",
    "LinearVPStochastic",
    "TrigonometricVPStochastic",
    "EncodingDecodingStochastic",
    "LowNoiseEncodingDecodingStochastic",
    "ConstantNoise",
    "OneSidedLinear",
    "OneSidedSBDM",
    "Coupling",
    "IndependentCoupling",
    "SinkhornCoupling",
    "ExactOTCoupling",
    "ExactConditionalCoupling",
    "C2OTCoupling",
    "StochasticField",
    "TwoSidedField",
    "TwoSidedField_v_etaz",
    "TwoSidedField_b_s",
    "TwoSidedField_x0_x1",
    "TwoSidedField_b_eta1",
    "OneSidedFlow",
    "OneSidedField_b",
    "OneSidedField_eta1",
    "field_factory",
    "make_default_net_constructor",
]
