# Reproduction order

Run the experiments in the following order.

1. Install the package and run `pytest`.
2. Download P-NS and LC through the tracked data downloaders.
3. Run `run_smoke.sh`.
4. Train and evaluate the five main methods with `run_main.sh`.
5. Train the P-NS `clawno` external baseline.
6. Generate and evaluate stress sets with `scripts/run_stress.py`.
7. Run the LC structural ablation with `scripts/run_ablation.py`.
8. Run tangent filtering, residual reset, and projection sensitivity diagnostics.
9. Benchmark end-to-end runtime.
10. Generate figures from the exported CSV files.

Do not select checkpoints or tune hyperparameters on the test or stress sets. The paper protocol uses validation rollout error for checkpoint selection and reports mean plus sample standard deviation over seeds 0, 1, and 2.
