"""Integration of learnable vector fields as ODEs (flows) and SDEs (processes).

``Flow`` integrates a deterministic velocity field and supports forward/backward
sampling as well as exact density evaluation via ``push_forward`` /
``push_backward``. ``Process`` integrates the corresponding SDE (diffusion).
See ``flow/README.md`` for the mathematical conventions.
"""

from flow.flow import Flow
from flow.process import Process
from flow.net import FourierFeatures, VelocityNet, DualOutputNet, EtaNet
from flow.integrators import (
    Integrator,
    Euler,
    RK4,
    EulerMaruyama,
    DiffraxIntegrator,
    get_integrator,
)
from flow.utils import divergence_jvp, divergence_hutchinson

__version__ = "0.3.0"

__all__ = [
    "Flow",
    "Process",
    "FourierFeatures",
    "VelocityNet",
    "DualOutputNet",
    "EtaNet",
    "Integrator",
    "Euler",
    "RK4",
    "EulerMaruyama",
    "DiffraxIntegrator",
    "get_integrator",
    "divergence_jvp",
    "divergence_hutchinson",
]
