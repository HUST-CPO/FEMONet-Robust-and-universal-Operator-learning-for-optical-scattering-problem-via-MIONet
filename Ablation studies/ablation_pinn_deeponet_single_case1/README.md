# Case 1: single-instance PINN-DeepONet

The operator PINN architecture and residual are reused on sample ID 1 (training row 449). All parameters remain trainable. The three arms use 1000 Adam updates, 3000 Adam updates, or 5000 Adam updates followed by up to 500 L-BFGS iterations. Each arm starts from the same seed-123 weights and uses its stated cosine-schedule horizon. Errors describe fitting on this instance.

## Run

Install the dependencies listed in [the study guide](../README.md). Run the commands below from this directory on a CUDA GPU.

```bash
python run_controlled.py --epochs 1000
python run_controlled.py --epochs 3000
python run_controlled.py --epochs 5000 --lbfgs 500
python compare_results.py
```

First complete `ablation_pinn_case1`; the runner checks its source, dataset, collocation, and initialization hashes. The stored full-operator same-instance prediction supplies the reference row.

## Outputs

New experiments write to `runs/`; comparison scripts write CSV tables and figures. Each run records settings, input hashes, losses, predictions, and error metrics. Use a fresh run name for each execution. `reference_results/` contains the recorded experiment metadata and tables supplied with this release; its hashes identify the original executed sources. New executions record the packaged source hashes.

## Metric

Complex field MSE is the sum of squared absolute errors divided by the number of valid degrees of freedom. Padding is excluded. SPP coefficient MSE and spatially integrated vector-field relative L2 are reported separately.
