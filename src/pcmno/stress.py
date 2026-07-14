from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch

from pcmno.data.lc import generate_lc_dataset
from pcmno.data.pns import generate_pns_split
from pcmno.evaluation import evaluate_checkpoint
from pcmno.utils import choose_device


def generate_stress_datasets(cfg: dict, output_root: str | Path) -> dict[str, Path]:
    """Generate the four paper stress configurations without model tuning."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    device = choose_device(cfg.get("runtime", {}).get("device", "auto"))
    generated: dict[str, Path] = {}

    pns_cfg = deepcopy(cfg["pns"])
    pns_count = int(cfg["stress"]["trajectories"])
    for reynolds in cfg["stress"]["pns_reynolds"]:
        path = root / "pns_re" / f"pns_re{int(reynolds)}.pt"
        generated[f"P-NS-Re-{int(reynolds)}"] = generate_pns_split(
            path,
            count=pns_count,
            config=pns_cfg,
            device=device,
            seed_offset=40_000 + int(reynolds),
            fixed_reynolds=float(reynolds),
        )

    grid_cfg = deepcopy(pns_cfg)
    grid_cfg["data_generation"]["resolution"] = int(cfg["stress"]["pns_grid_resolution"])
    grid_cfg["physics"]["resolution"] = int(cfg["stress"]["pns_grid_resolution"])
    path = root / "pns_grid" / f"pns_n{cfg['stress']['pns_grid_resolution']}.pt"
    generated["P-NS-Grid"] = generate_pns_split(
        path,
        count=pns_count,
        config=grid_cfg,
        device=device,
        seed_offset=50_000,
    )

    lc_cfg = deepcopy(cfg["lc"])
    for reynolds in cfg["stress"]["lc_reynolds"]:
        local = deepcopy(lc_cfg)
        local["splits"] = {"train": 0, "val": 0, "test": int(cfg["stress"]["trajectories"])}
        path = root / "lc_re" / f"lc_re{int(reynolds)}.npz"
        generated[f"LC-Re-{int(reynolds)}"] = generate_lc_dataset(
            path, local, device=device, fixed_reynolds=float(reynolds)
        )
    local = deepcopy(lc_cfg)
    local["splits"] = {"train": 0, "val": 0, "test": int(cfg["stress"]["trajectories"])}
    lid_value = float(cfg["stress"]["lc_lid_velocity"])
    path = root / "lc_lid" / f"lc_lid{lid_value:g}.npz"
    generated["LC-Lid"] = generate_lc_dataset(
        path, local, device=device, fixed_lid=lid_value
    )
    return generated


def evaluate_stress_manifest(
    manifest: dict[str, Path],
    pns_cfg: dict,
    lc_cfg: dict,
    checkpoints: dict[str, dict[str, str]],
    horizons=(50,),
) -> pd.DataFrame:
    rows = []
    for setting, path in manifest.items():
        cfg = pns_cfg if setting.startswith("P-NS") else lc_cfg
        split = "test"
        for method, checkpoint in checkpoints["pns" if setting.startswith("P-NS") else "lc"].items():
            curve, summary = evaluate_checkpoint(
                cfg,
                method=method,
                checkpoint=checkpoint,
                split=split,
                data_override=path,
                horizons=horizons,
            )
            summary["setting"] = setting
            summary["data_path"] = str(path)
            rows.append(summary)
    return pd.DataFrame(rows)
