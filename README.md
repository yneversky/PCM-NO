# Learning Neural Operators on Solver-Defined Physical Constraint Manifolds

This repository contains the reference implementation of **Physical-Constraint-Manifold Neural Operators (PCM-NO)**. PCM-NO constrains an autoregressive neural-operator transition using fixed layers derived from the discrete operators of a numerical solver:

\[
\Phi_\theta(u_{h,k};\mu,g_h,h)
=
R_h\!\left(
    u_{h,k}
    +
    \Delta t\,P_hK_\theta(u_{h,k},\mu,g_h,h);
    b_h
\right).
\]

The neural backbone predicts a candidate residual update. The tangent projection \(P_h\) removes the constraint-incompatible component before state advancement, and the metric retraction \(R_h\) returns the advanced state to the solver-defined affine feasible set. Only the backbone \(K_\theta\) is trainable.

The code supports the two configurations studied in the paper:

- **P-NS:** periodic forced two-dimensional flow with a spectral incompressibility constraint.
- **LC:** MAC-stored lid-driven cavity trajectories with joint incompressibility and prescribed-boundary constraints.

## Repository layout

```text
.
├── configs/                    # Paper, smoke, stress, and ablation protocols
├── data/
│   ├── pns/                    # P-NS downloader, metadata, checksum, sample
│   └── lc/                     # LC downloader, metadata, checksum, sample
├── notebooks/                 # Lightweight Colab entry points
├── reproduction/              # Ordered reproduction commands
├── scripts/                   # Training, evaluation, stress, diagnostics, plots
├── src/pcmno/
│   ├── data/                   # Released-data loaders and solver generators
│   ├── models/                 # FNO and matched ClawNO-style baseline
│   ├── operators/              # Spectral and MAC constraint operators
│   ├── diagnostics.py
│   ├── evaluation.py
│   ├── losses.py
│   ├── metrics.py
│   ├── stress.py
│   ├── training.py
│   └── transitions.py
└── tests/                      # CPU unit tests for operators and transitions
```

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/yneversky/PCM-NO.git
cd PCM-NO
python -m pip install --upgrade pip
pip install -e .
```

For development and unit tests:

```bash
pip install -e ".[dev]"
pytest
```

CUDA is used automatically when available. Set `runtime.device: cpu` in a YAML configuration, or pass an override such as `--set runtime.device=cpu`, to force CPU execution.

## Download the released datasets

The full datasets are distributed as GitHub Release assets and are not stored in Git history.

```bash
python data/pns/download_pns.py --repo yneversky/PCM-NO
python data/lc/download_lc.py --repo yneversky/PCM-NO
```

The downloaders place the files under:

```text
data/pns/raw/
data/lc/raw/
```

and verify them against the tracked `SHA256SUMS` manifests. Dataset layouts, solver settings, and loading examples are documented in [`data/pns/README.md`](data/pns/README.md) and [`data/lc/README.md`](data/lc/README.md).

The tracked sample files are intended only for loader checks and smoke runs. They must not be used to report paper results.

## Smoke test

After the small samples are present:

```bash
bash reproduction/run_smoke.sh
```

The smoke protocol trains a reduced PCM-NO model for one epoch on each sample dataset. It checks code paths and tensor compatibility rather than predictive performance.

## Main in-distribution experiments

### P-NS

Train one method and seed:

```bash
python scripts/train.py \
  --config configs/pns_paper.yaml \
  --method pcmno \
  --seed 0
```

Evaluate the validation-selected checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/pns_paper.yaml \
  --method pcmno \
  --seed 0
```

The supported P-NS methods are:

```text
fno, divreg, pino, finalproj, pcmno, clawno
```

`clawno` uses a scalar potential output followed by fixed spectral differentiation. It is an external structure-preserving comparison and is trained with the same autoregressive data loss and FNO-scale capacity.

### LC

```bash
python scripts/train.py \
  --config configs/lc_paper.yaml \
  --method pcmno \
  --seed 0

python scripts/evaluate.py \
  --config configs/lc_paper.yaml \
  --method pcmno \
  --seed 0
```

The supported LC main methods are:

```text
fno, divreg, pino, finalproj, pcmno
```

Run the complete three-seed main protocol with:

```bash
bash reproduction/run_main.sh
```

This command is computationally expensive. Checkpoint resume is enabled in the paper configurations.

## Output organization

The default main output tree is:

```text
outputs/main/
├── pns/
│   └── <method>/seed_<seed>/
│       ├── best.pt
│       ├── latest.pt
│       ├── history.csv
│       ├── resolved_config.json
│       └── evaluation/
│           ├── test_curve.csv
│           └── test_summary.json
└── lc/
    └── <method>/seed_<seed>/
        └── ...
```

`best.pt` is selected only from validation rollout error. Test data are not used for checkpoint selection.

## Evaluation metrics

The evaluators export complete rollout curves and the paper horizons \(r\in\{1,20,50\}\).

P-NS reports:

- relative state error `RelL2`;
- base-10 log normalized spectral divergence `LogDiv`;
- relative kinetic-energy error `EnergyRelErr`.

LC reports:

- relative state error `RelL2`;
- base-10 log interior MAC divergence `LogDiv`;
- base-10 log prescribed-boundary residual `LogBC`;
- centerline velocity-profile error `ProfileErr`.

Aggregate per-seed summaries with:

```bash
python scripts/aggregate_results.py \
  --summaries outputs/main/pns/pcmno/seed_*/evaluation/test_summary.json \
  --output outputs/main/pns_pcmno_summary.csv
```

## Stress evaluation

The stress suite covers:

- P-NS Reynolds-number extrapolation;
- LC Reynolds-number extrapolation;
- LC lid-velocity shift;
- P-NS target-grid transfer to \(128^2\).

Copy [`configs/checkpoints.example.json`](configs/checkpoints.example.json) to a local manifest and replace the paths with the desired validation-selected checkpoints. Then run:

```bash
python scripts/run_stress.py \
  --config configs/stress.yaml \
  --pns-config configs/pns_paper.yaml \
  --lc-config configs/lc_paper.yaml \
  --checkpoints checkpoints.json
```

Stress sets are generated from the same discrete solvers used for the released data. No stress-set tuning or test-time fine-tuning is performed.

## Component ablation

The LC ablation isolates the internal PCM-NO components:

- `tangent_only`: projected residual update without final state retraction;
- `retraction_only`: unconstrained residual update followed by retraction;
- `div_only`: divergence-only tangent projection and retraction, omitting prescribed-boundary constraints;
- `pcmno`: complete tangent projection and affine state retraction.

```bash
python scripts/run_ablation.py --config configs/lc_ablation.yaml
```

The public ablation configuration uses the exact separable DCT solve during training and validation and the paper CG solve for final evaluation. The full method can instead be referenced from the completed main experiment when reproducing the exact paper table.

## Mechanism diagnostics

### Tangent filtering

```bash
python scripts/run_diagnostics.py \
  --config configs/pns_paper.yaml \
  --checkpoint outputs/main/pns/pcmno/seed_0/best.pt \
  --diagnostic tangent \
  --output-dir outputs/diagnostics/pns
```

The exported statistics include raw and projected update errors, the removed normal-component energy fraction, and the normalized Pythagorean gap.

### Residual accumulation and reset

```bash
python scripts/run_diagnostics.py \
  --config configs/lc_paper.yaml \
  --checkpoint outputs/main/lc/pcmno/seed_0/best.pt \
  --diagnostic residual-reset \
  --output-dir outputs/diagnostics/lc
```

The two replay branches receive the same projected update sequence. `TangentOnlyReplay` omits the final state retraction, while `FullReplay` applies it after every step.

### Projection-solver sensitivity

```bash
python scripts/run_diagnostics.py \
  --config configs/lc_paper.yaml \
  --checkpoint outputs/main/lc/pcmno/seed_0/best.pt \
  --diagnostic projection-sensitivity \
  --output-dir outputs/diagnostics/lc

python scripts/plot_projection_sensitivity.py \
  --csv outputs/diagnostics/lc/lc_projection_sensitivity.csv \
  --output outputs/diagnostics/lc/lc_projection_sensitivity.pdf
```

## Runtime benchmark

Prepare a JSON mapping method names to checkpoints, then run:

```bash
python scripts/benchmark_runtime.py \
  --config configs/pns_paper.yaml \
  --checkpoints pns_checkpoints.json \
  --batch-size 1 \
  --batch-size 8 \
  --output outputs/pns_runtime.csv
```

The benchmark advances learned transitions and reference solvers over the same saved physical interval. It excludes checkpoint loading and file I/O.

## Figures

Long-rollout curves:

```bash
python scripts/make_figures.py long-rollout \
  --curves outputs/main/pns/*/seed_*/evaluation/test_curve.csv \
           outputs/main/lc/*/seed_*/evaluation/test_curve.csv \
  --output figures/generated/long_rollout.pdf
```

Qualitative stress comparison:

```bash
python scripts/make_qualitative.py \
  --config configs/pns_paper.yaml \
  --data outputs/stress/data/pns_re/pns_re1000.pt \
  --finalproj outputs/main/pns/finalproj/seed_0/best.pt \
  --pcmno outputs/main/pns/pcmno/seed_0/best.pt \
  --step 50 \
  --output figures/generated/pns_re1000_fields.pdf
```

The displayed trajectory is selected by median final-step ground-truth kinetic energy, independently of model error.

## Protocol notes

- P-NS uses a dealiased spectral Helmholtz projection for both tangent filtering and state retraction.
- LC uses the same MAC divergence, boundary lifting, free-face adjoint, and normal-equation projection in data handling, training, and evaluation.
- The released LC generator evolves diffusion and linear drag, then applies boundary enforcement and pressure projection. The public code follows that released generator exactly and does not add an unimplemented advection term.
- `DivReg` and `PINO` use physical terms only during training, so their inference graph is the ambient FNO graph.
- The structural propositions guarantee the selected solver-level constraints. They do not imply rollout accuracy, long-horizon stability, or full PDE correctness.

## Reproducibility scope

The repository contains training and evaluation code, solver-based data generation, stress protocols, component ablations, mechanism diagnostics, runtime measurement, and figure export. Full paper-scale reproduction requires the released datasets, sufficient GPU time, and the paper configurations. Pretrained checkpoints are not required for training from scratch and are not assumed to be present unless released separately.

## Citation

Please cite the PCM-NO paper when using this code or the released datasets. The final BibTeX entry will be added after the public preprint or proceedings record is available.
