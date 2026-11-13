# Case 1: label efficiency at matched updates

PI, supervised DeepONet (DD), and Hybrid use 39000 optimizer updates. Label counts are 10, 30, 100, 300, and 1000. PI and Hybrid physics batches use all 1246 training conditions. Supervised batches use a cyclic shuffled stream of 32 labels.

## Run

Install the dependencies listed in [the study guide](../README.md). Run the commands below from this directory on a CUDA GPU.

```bash
python run_experiment.py --mode verify --run-name verify
python run_experiment.py --mode pi --run-name pi_n0_seed123
for n in 10 30 100 300 1000; do
  python run_experiment.py --mode dd --n-label "$n" --run-name "dd_n${n}_seed123"
  python run_experiment.py --mode hybrid --n-label "$n" --run-name "hybrid_n${n}_seed123"
done
python compare_results.py
```

## Outputs

New experiments write to `runs/`; comparison scripts write CSV tables and figures. Each run records settings, input hashes, losses, predictions, and error metrics. Use a fresh run name for each execution. `reference_results/` contains the recorded experiment metadata and tables supplied with this release; its hashes identify the original executed sources. New executions record the packaged source hashes.

## Metric

Complex field MSE is the sum of squared absolute errors divided by the number of valid degrees of freedom. Padding is excluded. SPP coefficient MSE and spatially integrated vector-field relative L2 are reported separately.
