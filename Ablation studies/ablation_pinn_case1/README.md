# Case 1: strong-form PINN-DeepONet

The CNN branch and smooth coordinate trunk predict the complex scattered field. The loss is the mean squared Helmholtz residual plus the mean squared Robin residual, with boundary weight 1. Triangle centroids and exterior-edge midpoints supply fixed collocation points. Coordinates are in micrometres and k = 2*pi/1.55. Training uses seed 123, 1000 epochs, batch size 32, and microbatches of 4.

## Run

Install the dependencies listed in [the study guide](../README.md). Run the commands below from this directory on a CUDA GPU.

```bash
python prepare_collocation.py
python run_pinn.py --mode verify --run-name verify
python run_pinn.py --mode train --run-name pinn_seed123
python compare_results.py
python plot_fields.py
```

First complete the Case 1 ILU comparison; its manifest supplies the paired initialization check. The included collocation arrays can be used directly; `prepare_collocation.py` regenerates them from the included mesh data.

## Outputs

New experiments write to `runs/`; comparison scripts write CSV tables and figures. Each run records settings, input hashes, losses, predictions, and error metrics. Use a fresh run name for each execution. `reference_results/` contains the recorded experiment metadata and tables supplied with this release; its hashes identify the original executed sources. New executions record the packaged source hashes.

## Metric

Complex field MSE is the sum of squared absolute errors divided by the number of valid degrees of freedom. Padding is excluded. SPP coefficient MSE and spatially integrated vector-field relative L2 are reported separately.
