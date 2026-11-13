# Case 1: coordinate PINN controls

Sample ID 1 (training row 449) uses a 2-64x6-2 tanh network in float64. Modes: manufactured solution, fixed collocation, resampled normalized residual with boundary weight 1 or 10, and ILU-FEM. Each starts from seed 123 and uses 5000 Adam updates followed by up to 500 L-BFGS iterations. Extended runs use up to 3000 L-BFGS iterations. Reports include final and independently physics-selected checkpoints. Errors describe fitting on this instance.

## Run

Install the dependencies listed in [the study guide](../README.md). Run the commands below from this directory on a CUDA GPU.

```bash
for mode in manufactured fixed resample1 resample10 ilu; do
  python run_single.py --mode "$mode"
done
python report_results.py
for mode in fixed resample1 ilu; do
  python run_single.py --mode "$mode" --lbfgs 3000 --tag extended
done
python report_extended.py
python audit_source.py
```

## Outputs

New experiments write to `runs/`; comparison scripts write CSV tables and figures. Each run records settings, input hashes, losses, predictions, and error metrics. Use a fresh run name for each execution. `reference_results/` contains the recorded experiment metadata and tables supplied with this release; its hashes identify the original executed sources. New executions record the packaged source hashes.

## Metric

Complex field MSE is the sum of squared absolute errors divided by the number of valid degrees of freedom. Padding is excluded. SPP coefficient MSE and spatially integrated vector-field relative L2 are reported separately.
