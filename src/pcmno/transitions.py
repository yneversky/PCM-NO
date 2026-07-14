from __future__ import annotations

from dataclasses import dataclass

import torch

from pcmno.operators.mac import mac_project, mac_project_div_only
from pcmno.operators.periodic import PeriodicSpectralOps


PNS_METHOD_ALIASES = {
    "fno": "fno",
    "divreg": "divreg",
    "divloss": "divreg",
    "pino": "pino",
    "pdeloss": "pino",
    "finalproj": "finalproj",
    "pcmno": "pcmno",
    "pcm_fno": "pcmno",
    "clawno": "clawno",
}

LC_METHOD_ALIASES = {
    "fno": "fno",
    "divreg": "divreg",
    "pino": "pino",
    "finalproj": "finalproj",
    "pcmno": "pcmno",
    "tangent_only": "tangent_only",
    "retraction_only": "retraction_only",
    "div_only": "div_only",
    "pcmno_no_boundary": "div_only",
}


@dataclass
class ProjectionSettings:
    train_iterations: int = 30
    eval_iterations: int = 120
    tolerance: float = 1e-6
    damping: float = 1e-8
    train_solver: str = "cg"
    eval_solver: str = "cg"

    def iterations(self, training: bool) -> int:
        return self.train_iterations if training else self.eval_iterations

    def solver(self, training: bool) -> str:
        return self.train_solver if training else self.eval_solver


class PNSDynamics:
    def __init__(self, operators: PeriodicSpectralOps, saved_interval: float = 0.05):
        self.operators = operators
        self.saved_interval = float(saved_interval)

    def step(
        self,
        method: str,
        model: torch.nn.Module,
        state: torch.Tensor,
        viscosity: torch.Tensor,
        training: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del training
        method = PNS_METHOD_ALIASES.get(method, method)
        raw = model(state, viscosity)
        auxiliary = {"raw": raw}
        if method == "pcmno":
            tangent = self.operators.project_velocity(raw)
            prediction = self.operators.project_velocity(
                state + self.saved_interval * tangent
            )
            auxiliary["tangent"] = tangent
            return prediction, auxiliary
        if method == "finalproj":
            return self.operators.project_velocity(raw), auxiliary
        if method in {"fno", "divreg", "pino", "clawno"}:
            return raw, auxiliary
        raise ValueError(f"Unknown P-NS method: {method}")


class LCDynamics:
    def __init__(self, projection: ProjectionSettings):
        self.projection = projection

    def _project(
        self,
        state: torch.Tensor,
        lid: torch.Tensor | None,
        zero_boundary_update: bool,
        training: bool,
    ) -> torch.Tensor:
        return mac_project(
            state,
            lid=lid,
            zero_boundary_update=zero_boundary_update,
            iterations=self.projection.iterations(training),
            tolerance=self.projection.tolerance,
            damping=self.projection.damping,
            solver=self.projection.solver(training),
        )

    def _project_div_only(self, state: torch.Tensor, training: bool) -> torch.Tensor:
        return mac_project_div_only(
            state,
            iterations=self.projection.iterations(training),
            tolerance=self.projection.tolerance,
            damping=self.projection.damping,
            solver=self.projection.solver(training),
        )

    def step(
        self,
        method: str,
        model: torch.nn.Module,
        state: torch.Tensor,
        lid: torch.Tensor,
        viscosity: torch.Tensor,
        training: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        method = LC_METHOD_ALIASES.get(method, method)
        raw = model(state, lid, viscosity)
        auxiliary = {"raw": raw}
        if method in {"fno", "divreg", "pino"}:
            return raw, auxiliary
        if method == "finalproj":
            return self._project(raw, lid, False, training), auxiliary
        if method == "pcmno":
            tangent = self._project(raw, None, True, training)
            prediction = self._project(state + tangent, lid, False, training)
            auxiliary["tangent"] = tangent
            return prediction, auxiliary
        if method == "tangent_only":
            tangent = self._project(raw, None, True, training)
            auxiliary["tangent"] = tangent
            return state + tangent, auxiliary
        if method == "retraction_only":
            return self._project(state + raw, lid, False, training), auxiliary
        if method == "div_only":
            tangent = self._project_div_only(raw, training)
            prediction = self._project_div_only(state + tangent, training)
            auxiliary["tangent"] = tangent
            return prediction, auxiliary
        raise ValueError(f"Unknown LC method: {method}")
