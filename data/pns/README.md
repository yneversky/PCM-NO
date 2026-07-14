# P-NS dataset

## Overview

P-NS contains forced periodic two-dimensional incompressible-flow trajectories
used by PCM-NO. The physical domain is the periodic square
\([0,2\pi]^2\). Each stored state is a two-component velocity field on a
\(64\times64\) grid.

The full dataset is distributed through the GitHub Release tag
`pns-data-v1.0.0`. The Git repository contains metadata, a downloader, and a
small sample, while the full binary files remain outside Git history.

## Release assets

| File | Trajectories | Velocity tensor shape |
|---|---:|---|
| `train.pt` | 500 | `[500, 56, 2, 64, 64]` |
| `val.pt` | 100 | `[100, 56, 2, 64, 64]` |
| `test.pt` | 200 | `[200, 56, 2, 64, 64]` |
| `SHA256SUMS` | — | Integrity manifest |

Each `.pt` file is a PyTorch dictionary with:

- `u`: velocity trajectories with layout
  `[trajectory, time, component, x, y]`;
- `Re`: one Reynolds number per trajectory;
- `nu`: one viscosity per trajectory, with `nu = 1 / Re`;
- `cfg`: the data-generation configuration.

Velocity component `0` is \(u_x\), and component `1` is \(u_y\).

## Generation settings

The paper-scale dataset uses:

- Reynolds number sampled uniformly from \([100,500]\);
- Kolmogorov forcing \(f=(0.1\sin(4y),0)\);
- linear drag coefficient `0.1`;
- a dealiased pseudo-spectral vorticity solver with the `2/3` rule;
- classical RK4 time integration;
- solver step `0.005` and stored-state interval `0.05`;
- eight stored intervals of burn-in;
- 56 stored states per trajectory.

The complete machine-readable configuration is in
[`pns_paper_config.json`](pns_paper_config.json).

## Download

From a clone of the repository:

```bash
python data/pns/download_pns.py --repo OWNER/REPO
```

The files are placed in `data/pns/raw/` by default. The script can infer the
repository from the Git remote, so `--repo` may be omitted inside a standard
clone. It also accepts the environment variable `PCMNO_GITHUB_REPO`.

A custom destination can be supplied with:

```bash
python data/pns/download_pns.py \
  --repo OWNER/REPO \
  --output-dir /path/to/pns
```

The downloader verifies every file against `SHA256SUMS`. Verification should
not be disabled for paper reproduction.

## Loading

```python
import torch

try:
    data = torch.load(
        "data/pns/raw/train.pt",
        map_location="cpu",
        weights_only=False,
    )
except TypeError:
    data = torch.load("data/pns/raw/train.pt", map_location="cpu")

u = data["u"]
Re = data["Re"]
nu = data["nu"]

print(u.shape)   # torch.Size([500, 56, 2, 64, 64])
print(Re.shape)  # torch.Size([500])
print(nu.shape)  # torch.Size([500])
```

## Small sample

`sample/pns_sample.pt` contains the first two trajectories from the training
split and preserves the same keys and tensor layout. It is intended only for
loader tests and smoke runs, not for reporting model performance.

Create the sample and the final checksum manifest directly from the Drive data:

```bash
python prepare_data_assets.py
```

The default source directory is:

```text
/content/drive/MyDrive/PCM_NO/exp1_dataset1/data/kolmogorov_N64_T56_dt0.05_sub10_Re100-500_kf4_trainH4_preset-paper_projfix_tuned
```

Use `--pns-source-dir` to override it.

## Integrity verification

After downloading:

```bash
cd data/pns/raw
sha256sum -c ../SHA256SUMS
```

Expected output contains `train.pt: OK`, `val.pt: OK`, and `test.pt: OK`.

## License and citation

Use the repository-level dataset license. Before public release, ensure that the
release page and repository specify the intended data license. Please cite the
PCM-NO paper when using this dataset; final bibliographic metadata should be
added after publication.
