# Supplementary SPP: ILU preconditioning

SPP1; raw FEM and ILU-FEM use the saved split, seed 123, and the archived MIONet settings.

## Run

Install the dependencies listed in [the study guide](../README.md). Run the commands below from this directory on a CUDA GPU.

```bash
python run_ablation.py --mode verify --run-name verify
python run_ablation.py --mode raw --run-name raw_seed123
python run_ablation.py --mode ilu --run-name ilu_seed123
python compare_results.py
```

## Outputs

New experiments write to `runs/`; comparison scripts write CSV tables and figures. Each run records settings, input hashes, losses, predictions, and error metrics. Use a fresh run name for each execution. `reference_results/` contains the recorded experiment metadata and tables supplied with this release; its hashes identify the original executed sources. New executions record the packaged source hashes.

## Metric

Complex field MSE is the sum of squared absolute errors divided by the number of valid degrees of freedom. Padding is excluded. SPP coefficient MSE and spatially integrated vector-field relative L2 are reported separately.
