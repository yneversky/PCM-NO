#!/usr/bin/env python3
"""Create the public sample files and SHA256SUMS from the final Drive datasets.

Run this script from the repository root after copying this package into the
repository. It does not copy the full datasets into Git; it only creates the
small sample files and checksum manifests used by the GitHub Release workflow.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any

DEFAULT_PNS_DIR = Path(
    "/content/drive/MyDrive/PCM_NO/exp1_dataset1/data/"
    "kolmogorov_N64_T56_dt0.05_sub10_Re100-500_kf4_"
    "trainH4_preset-paper_projfix_tuned"
)
DEFAULT_LC_FILE = Path(
    "/content/drive/MyDrive/PCMNO_Dataset2_MAC/"
    "mac_solver_table1/paper_N64/data/"
    "mac_cavity_id_paper_N64_T56_Re100-500_lid0.8-1.2.npz"
)
LC_FILENAME = DEFAULT_LC_FILE.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create P-NS/LC sample files and release checksum manifests."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--pns-source-dir", type=Path, default=DEFAULT_PNS_DIR)
    parser.add_argument("--lc-source-file", type=Path, default=DEFAULT_LC_FILE)
    parser.add_argument("--sample-trajectories", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-pns", action="store_true")
    parser.add_argument("--skip-lc", action="store_true")
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{sha256(path)}  {path.name}" for path in paths]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {output}")
    for line in lines:
        print(f"        {line}")


def torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def prepare_pns(repo_root: Path, source_dir: Path, n_sample: int, overwrite: bool) -> None:
    import torch

    split_paths = [source_dir / "train.pt", source_dir / "val.pt", source_dir / "test.pt"]
    for path in split_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    expected_counts = {"train.pt": 500, "val.pt": 100, "test.pt": 200}
    for path in split_paths:
        obj = torch_load(path)
        required = {"u", "Re", "nu", "cfg"}
        missing = required.difference(obj)
        if missing:
            raise KeyError(f"{path} is missing keys: {sorted(missing)}")
        u = obj["u"]
        expected_shape = (expected_counts[path.name], 56, 2, 64, 64)
        if tuple(u.shape) != expected_shape:
            raise ValueError(f"Unexpected {path.name} u.shape={tuple(u.shape)}; expected {expected_shape}")
        if tuple(obj["Re"].shape) != (expected_shape[0],):
            raise ValueError(f"Unexpected Re shape in {path}")
        if tuple(obj["nu"].shape) != (expected_shape[0],):
            raise ValueError(f"Unexpected nu shape in {path}")
        print(f"[validated] {path.name}: u{tuple(u.shape)}")

    train = torch_load(split_paths[0])
    if not 1 <= n_sample <= train["u"].shape[0]:
        raise ValueError("--sample-trajectories must be between 1 and the train split size")

    output = repo_root / "data" / "pns" / "sample" / "pns_sample.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")

    sample = {
        "u": train["u"][:n_sample].contiguous(),
        "Re": train["Re"][:n_sample].contiguous(),
        "nu": train["nu"][:n_sample].contiguous(),
        "cfg": train["cfg"],
        "source_split": "train",
        "source_indices": torch.arange(n_sample, dtype=torch.long),
    }
    torch.save(sample, output)
    print(f"[write] {output}: u{tuple(sample['u'].shape)}")

    write_checksums(split_paths, repo_root / "data" / "pns" / "SHA256SUMS")


def prepare_lc(repo_root: Path, source_file: Path, n_sample: int, overwrite: bool) -> None:
    import numpy as np

    if not source_file.is_file():
        raise FileNotFoundError(source_file)

    required = {
        "states", "lid", "Re", "nu", "idx_train", "idx_val", "idx_test",
        "N", "S", "saved_steps", "inner_steps", "dt", "dt_save",
    }
    with np.load(source_file, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{source_file} is missing keys: {sorted(missing)}")
        expected_shape = (800, 56, 2, 66, 66)
        if tuple(data["states"].shape) != expected_shape:
            raise ValueError(
                f"Unexpected states.shape={tuple(data['states'].shape)}; expected {expected_shape}"
            )
        if tuple(data["idx_train"].shape) != (500,):
            raise ValueError("Unexpected idx_train shape")
        if tuple(data["idx_val"].shape) != (100,):
            raise ValueError("Unexpected idx_val shape")
        if tuple(data["idx_test"].shape) != (200,):
            raise ValueError("Unexpected idx_test shape")
        source_indices = data["idx_train"][:n_sample].astype(np.int64, copy=True)
        if not 1 <= n_sample <= data["idx_train"].shape[0]:
            raise ValueError("--sample-trajectories must be between 1 and the train split size")

        output = repo_root / "data" / "lc" / "sample" / "lc_sample.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and not overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")

        np.savez_compressed(
            output,
            states=data["states"][source_indices],
            lid=data["lid"][source_indices],
            Re=data["Re"][source_indices],
            nu=data["nu"][source_indices],
            source_indices=source_indices,
            source_split=np.asarray("train"),
            N=data["N"],
            S=data["S"],
            saved_steps=data["saved_steps"],
            inner_steps=data["inner_steps"],
            dt=data["dt"],
            dt_save=data["dt_save"],
        )
        print(f"[write] {output}: states{tuple(data['states'][source_indices].shape)}")

    write_checksums([source_file], repo_root / "data" / "lc" / "SHA256SUMS")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if args.sample_trajectories < 1:
        raise ValueError("--sample-trajectories must be positive")
    if not args.skip_pns:
        prepare_pns(
            repo_root,
            args.pns_source_dir.expanduser().resolve(),
            args.sample_trajectories,
            args.overwrite,
        )
    if not args.skip_lc:
        prepare_lc(
            repo_root,
            args.lc_source_file.expanduser().resolve(),
            args.sample_trajectories,
            args.overwrite,
        )
    print("Dataset release metadata and samples are ready.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
