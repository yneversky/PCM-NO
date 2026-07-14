import math

import torch

from pcmno.models import PNSFNO2d
from pcmno.operators.periodic import PeriodicSpectralOps
from pcmno.transitions import PNSDynamics


def test_periodic_projection_reduces_divergence():
    torch.manual_seed(0)
    operators = PeriodicSpectralOps(16, device="cpu")
    raw = torch.randn(4, 2, 16, 16)
    projected = operators.project_velocity(raw)
    before = operators.divergence(raw).square().mean().sqrt()
    after = operators.divergence(projected).square().mean().sqrt()
    assert torch.isfinite(projected).all()
    assert after < before * 1e-4


def test_pns_pcmno_transition_is_feasible():
    torch.manual_seed(1)
    operators = PeriodicSpectralOps(16, length=2 * math.pi, device="cpu")
    dynamics = PNSDynamics(operators, saved_interval=0.05)
    model = PNSFNO2d(modes=4, width=8, layers=2)
    state = operators.project_velocity(torch.randn(2, 2, 16, 16))
    viscosity = torch.tensor([0.01, 0.005])
    prediction, _ = dynamics.step("pcmno", model, state, viscosity)
    residual = operators.divergence(prediction).square().mean().sqrt()
    assert prediction.shape == state.shape
    assert residual < 1e-4
