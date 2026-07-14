from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pcmno.data import LCWindowDataset, PNSWindowDataset
from pcmno.evaluation import evaluate_lc, evaluate_pns
from pcmno.factory import build_dynamics, build_model, dataset_name, dataset_paths
from pcmno.losses import lc_rollout_loss, pns_rollout_loss
from pcmno.utils import (
    append_csv,
    atomic_torch_save,
    choose_device,
    count_parameters,
    safe_torch_load,
    seed_all,
    write_json,
)


def _loader(dataset, batch_size: int, shuffle: bool, seed: int, drop_last: bool):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def _checkpoint_paths(cfg: dict, method: str, seed: int, output_root: Path):
    run = output_root / dataset_name(cfg) / method / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    return run, run / "latest.pt", run / "best.pt", run / "history.csv"


def _save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_score: float,
    cfg: dict,
    method: str,
    seed: int,
):
    atomic_torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "best_score": float(best_score),
            "config": cfg,
            "method": method,
            "seed": int(seed),
        },
        path,
    )


def _validation_score(cfg: dict, model, dynamics, method: str, paths, device):
    dataset = dataset_name(cfg)
    validation_horizon = int(cfg["training"].get("validation_horizon", 20))
    if dataset == "pns":
        curve, _ = evaluate_pns(
            model,
            method,
            cfg,
            paths["val"],
            device,
            horizons=[validation_horizon],
            batch_size=int(cfg["evaluation"]["batch_size"]),
        )
    else:
        curve, _ = evaluate_lc(
            model,
            method,
            cfg,
            paths["archive"],
            "val",
            device,
            horizons=[validation_horizon],
            batch_size=int(cfg["evaluation"]["batch_size"]),
        )
    return float(curve.iloc[-1]["RelL2"]), curve.iloc[-1].to_dict()


def train(cfg: dict, method: str, seed: int, output_root: str | Path | None = None) -> Path:
    seed_all(seed, bool(cfg.get("runtime", {}).get("deterministic", False)))
    device = choose_device(cfg.get("runtime", {}).get("device", "auto"))
    paths = dataset_paths(cfg)
    output_root = Path(output_root or cfg["output"]["root"]).expanduser().resolve()
    run, latest_path, best_path, history_path = _checkpoint_paths(
        cfg, method, seed, output_root
    )
    write_json(cfg, run / "resolved_config.json")
    model_data_path = paths.get("archive", paths.get("train"))
    model = build_model(cfg, method, device, data_path=model_data_path)
    dynamics = build_dynamics(cfg, device)
    training_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(training_cfg["epochs"]))
    )
    start_epoch = 1
    best_score = float("inf")
    if bool(training_cfg.get("resume", True)) and latest_path.exists():
        checkpoint = safe_torch_load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", best_score))

    dataset = dataset_name(cfg)
    horizon = int(training_cfg["rollout_horizon"])
    if dataset == "pns":
        train_dataset = PNSWindowDataset(paths["train"], horizon=horizon, stride=1)
    else:
        train_dataset = LCWindowDataset(paths["archive"], "train", horizon=horizon, stride=1)
    loader = _loader(
        train_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    patience = training_cfg.get("early_stopping_patience")
    non_improving = 0
    for epoch in range(start_epoch, int(training_cfg["epochs"]) + 1):
        model.train()
        start = time.perf_counter()
        total_loss = 0.0
        examples = 0
        for batch in tqdm(loader, desc=f"{dataset}/{method}/seed{seed}/epoch{epoch}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            if dataset == "pns":
                sequence, viscosity = batch
                sequence = sequence.to(device, non_blocking=True)
                viscosity = viscosity.to(device, non_blocking=True)
                loss, _ = pns_rollout_loss(
                    method,
                    model,
                    sequence,
                    viscosity,
                    dynamics,
                    cfg["loss"],
                )
                batch_size = sequence.shape[0]
            else:
                sequence, lid, viscosity = batch
                sequence = sequence.to(device, non_blocking=True)
                lid = lid.to(device, non_blocking=True)
                viscosity = viscosity.to(device, non_blocking=True)
                loss, _ = lc_rollout_loss(
                    method,
                    model,
                    sequence,
                    lid,
                    viscosity,
                    dynamics,
                    cfg["loss"],
                    saved_interval=float(cfg["physics"]["saved_interval"]),
                    drag=float(cfg["physics"]["drag"]),
                )
                batch_size = sequence.shape[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_cfg.get("gradient_clip", 1.0))
            )
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * batch_size
            examples += batch_size
        scheduler.step()
        validate_every = int(training_cfg.get("validate_every", 1))
        validation_score = float("nan")
        validation_metrics = {}
        improved = False
        if epoch == 1 or epoch % validate_every == 0 or epoch == int(training_cfg["epochs"]):
            validation_score, validation_metrics = _validation_score(
                cfg, model, dynamics, method, paths, device
            )
            improved = validation_score < best_score
            if improved:
                best_score = validation_score
                non_improving = 0
                _save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_score,
                    cfg,
                    method,
                    seed,
                )
            else:
                non_improving += 1
        elapsed = time.perf_counter() - start
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(examples, 1),
            "validation_score": validation_score,
            "best_score": best_score,
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": elapsed,
            "parameters": count_parameters(model),
            **{f"val_{key}": value for key, value in validation_metrics.items()},
        }
        append_csv(row, history_path)
        _save_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_score,
            cfg,
            method,
            seed,
        )
        print(
            f"epoch={epoch:03d} train={row['train_loss']:.6e} "
            f"val={validation_score:.6e} best={best_score:.6e} sec={elapsed:.1f}"
        )
        if patience is not None and non_improving >= int(patience):
            print(f"Early stopping after {non_improving} non-improving validations.")
            break
    if not best_path.exists():
        _save_checkpoint(
            best_path,
            model,
            optimizer,
            scheduler,
            max(start_epoch - 1, 0),
            best_score,
            cfg,
            method,
            seed,
        )
    return best_path


def aggregate_histories(output_root: str | Path, dataset: str, method: str) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(output_root).glob(f"{dataset}/{method}/seed_*/history.csv")):
        frame = pd.read_csv(path)
        frame["seed"] = int(path.parent.name.split("_")[-1])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
