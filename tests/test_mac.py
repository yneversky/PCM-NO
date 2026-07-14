import torch

from pcmno.models import LCFNO2d
from pcmno.operators.mac import (
    apply_mac_boundary,
    mac_boundary_residual,
    mac_divergence,
    mac_project,
)
from pcmno.transitions import LCDynamics, ProjectionSettings


def test_mac_projection_enforces_boundary_and_reduces_divergence():
    torch.manual_seed(0)
    state = torch.randn(3, 2, 10, 10)
    lid = torch.tensor([0.8, 1.0, 1.2])
    bounded = apply_mac_boundary(state, lid)
    projected = mac_project(
        state,
        lid=lid,
        iterations=80,
        damping=1e-8,
        solver="cg",
    )
    before = mac_divergence(bounded).square().mean().sqrt()
    after = mac_divergence(projected).square().mean().sqrt()
    assert after < before
    assert mac_boundary_residual(projected, lid).max() < 2e-6


def test_dct_projection_is_finite():
    torch.manual_seed(1)
    state = torch.randn(2, 2, 10, 10)
    lid = torch.tensor([1.0, 1.1])
    projected = mac_project(state, lid=lid, solver="dct", damping=1e-8)
    assert torch.isfinite(projected).all()
    assert mac_boundary_residual(projected, lid).max() < 2e-6


def test_lc_pcmno_transition_shapes_and_boundaries():
    torch.manual_seed(2)
    model = LCFNO2d(stored_size=10, modes=3, width=8, layers=2)
    dynamics = LCDynamics(
        ProjectionSettings(
            train_iterations=20,
            eval_iterations=40,
            tolerance=1e-6,
            damping=1e-8,
        )
    )
    lid = torch.tensor([0.9, 1.1])
    viscosity = torch.tensor([0.01, 0.005])
    state = mac_project(torch.randn(2, 2, 10, 10), lid=lid, iterations=60)
    prediction, _ = dynamics.step("pcmno", model, state, lid, viscosity)
    assert prediction.shape == state.shape
    assert mac_boundary_residual(prediction, lid).max() < 2e-6
