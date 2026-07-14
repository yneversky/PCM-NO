# LC dataset

## Overview

LC contains two-dimensional lid-driven cavity velocity trajectories used by
PCM-NO. The domain is the unit square \([0,1]^2\), with no-slip walls and a
prescribed moving lid. The active MAC grid has resolution \(64\times64\).
Each velocity component is stored in a padded \(66\times66\) array containing
boundary, ghost, and unused padding entries required by the discrete operators.

The full dataset is distributed through the GitHub Release tag
`lc-data-v1.0.0`. The Git repository contains metadata, a downloader, and a
small sample, while the full archive remains outside Git history.

## Release asset

```text
mac_cavity_id_paper_N64_T56_Re100-500_lid0.8-1.2.npz
```

The principal array has shape:

```text
states: [800, 56, 2, 66, 66]
```

Its layout is `[trajectory, time, component, row, column]`. Component `0` is the
horizontal velocity `u`, and component `1` is the vertical velocity `v`.

The NPZ archive contains:

- `states`: velocity trajectories;
- `lid`: one prescribed lid velocity per trajectory;
- `Re`: one Reynolds number per trajectory;
- `nu`: one viscosity per trajectory, with `nu = 1 / Re`;
- `idx_train`, `idx_val`, `idx_test`: fixed split indices of sizes 500, 100,
  and 200;
- `N`, `S`, `saved_steps`, `inner_steps`, `dt`, and `dt_save`: discretization
  metadata.

## Generation settings

The paper-scale dataset uses:

- Reynolds number sampled uniformly from \([100,500]\);
- lid velocity sampled uniformly from \([0.8,1.2]\);
- active grid resolution \(64\times64\), stored as \(66\times66\);
- solver time step `0.0006` and stored-state interval `0.006`;
- four stored intervals of burn-in;
- 56 stored states per trajectory;
- a MAC-style projected velocity solver with 160 data-generation projection
  iterations, tolerance `1e-7`, and damping `1e-8`;
- linear drag coefficient `0.03` and initial noise scale `0.02`.

The complete machine-readable configuration is in
[`lc_paper_config.json`](lc_paper_config.json).

## Download

From a clone of the repository:

```bash
python data/lc/download_lc.py --repo OWNER/REPO
```

The archive is placed in `data/lc/raw/` by default. The script can infer the
repository from the Git remote, so `--repo` may be omitted inside a standard
clone. It also accepts the environment variable `PCMNO_GITHUB_REPO`.

A custom destination can be supplied with:

```bash
python data/lc/download_lc.py \
  --repo OWNER/REPO \
  --output-dir /path/to/lc
```

The downloader verifies the archive against `SHA256SUMS`.

## Loading and reproducing the paper split

```python
import numpy as np

path = (
    "data/lc/raw/"
    "mac_cavity_id_paper_N64_T56_Re100-500_lid0.8-1.2.npz"
)

with np.load(path, allow_pickle=False) as data:
    train_states = data["states"][data["idx_train"]]
    val_states = data["states"][data["idx_val"]]
    test_states = data["states"][data["idx_test"]]

    train_lid = data["lid"][data["idx_train"]]
    train_Re = data["Re"][data["idx_train"]]
    train_nu = data["nu"][data["idx_train"]]

print(train_states.shape)  # (500, 56, 2, 66, 66)
print(val_states.shape)    # (100, 56, 2, 66, 66)
print(test_states.shape)   # (200, 56, 2, 66, 66)
```

## Small sample

`sample/lc_sample.npz` contains the first two trajectories selected by
`idx_train` and preserves the state layout and scalar discretization metadata.
It is intended only for loader tests and smoke runs.

Create the sample and final checksum manifest directly from the Drive data:

```bash
python prepare_data_assets.py
```

The default source file is:

```text
/content/drive/MyDrive/PCMNO_Dataset2_MAC/mac_solver_table1/paper_N64/data/mac_cavity_id_paper_N64_T56_Re100-500_lid0.8-1.2.npz
```

Use `--lc-source-file` to override it.

## Integrity verification

After downloading:

```bash
cd data/lc/raw
sha256sum -c ../SHA256SUMS
```

The expected output contains the dataset filename followed by `OK`.

## License and citation

Use the repository-level dataset license. Before public release, ensure that the
release page and repository specify the intended data license. Please cite the
PCM-NO paper when using this dataset; final bibliographic metadata should be
added after publication.
