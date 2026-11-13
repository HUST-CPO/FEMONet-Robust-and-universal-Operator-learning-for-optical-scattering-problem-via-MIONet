# Ablation studies

Reference implementations for preconditioning, label efficiency, and physics-informed baselines accompanying FEMONet.

## Repository layout

| Folder | Experiment |
| --- | --- |
| [ablation_no_ilu_case1](ablation_no_ilu_case1/README.md) | Case 1: ILU preconditioning |
| [ablation_no_ilu_case4](ablation_no_ilu_case4/README.md) | Case 4: ILU preconditioning |
| [ablation_no_ilu_s2](ablation_no_ilu_s2/README.md) | Supplementary SPP: ILU preconditioning |
| [ablation_label_efficiency_case1](ablation_label_efficiency_case1/README.md) | Case 1: label efficiency at matched updates |
| [ablation_label_efficiency_case1_epochs](ablation_label_efficiency_case1_epochs/README.md) | Case 1: label efficiency at matched epochs |
| [ablation_pinn_case1](ablation_pinn_case1/README.md) | Case 1: strong-form PINN-DeepONet |
| [ablation_pinn_single_case1](ablation_pinn_single_case1/README.md) | Case 1: coordinate PINN controls |
| [ablation_pinn_deeponet_single_case1](ablation_pinn_deeponet_single_case1/README.md) | Case 1: single-instance PINN-DeepONet |
| [non_permittivity](non_permittivity/README.md) | Three permittivity-input reference cases |

The existing Case 1 `VPINN single acse`, `Modified VPINN single acse`, and `VPINN-DeepOnet` directories contain the weak-form baselines.

## Requirements

Python 3.10 or later, PyTorch 2.5.1, NumPy, SciPy, h5py, Matplotlib, pandas, and tqdm. Install from the repository's `requirements-gpu.txt` for the recorded GPU environment. MATLAB is used for SPP field reconstruction. Training runners use CUDA and float64/complex128.

```bash
python -m pip install -r "Ablation studies/requirements.txt"
python "Ablation studies/check_release.py"
```

## Reproduction order

1. Run the three ILU comparisons, including their verification stages.
2. Run either or both label-efficiency protocols and their full-subset extension.
3. Run the Case 1 strong-form PINN-DeepONet comparison.
4. Run coordinate-only and PINN-DeepONet single-instance controls.
5. Run the comparison and plotting commands in each study README.

All operator Case 1 experiments use the included `A-proceeding/deepOnet_data_A1_all.mat` and its 1246/312 split. Input directories are resolved relative to the release. Runners accepting `--source` support an alternative directory with the same files. The bundled model reproduces the archived Case 1 test MSE of 4.390900391795585e-5.

The label-efficiency studies use nested subsets, seed 123, Adam with learning rate 1e-3, cosine decay to 1e-5, clipping norm 1, and unit Hybrid weights. The matched-update and matched-epoch protocols have separate directories and reports. Final operator checkpoints are evaluated on the fixed test set. Single-instance reports identify their optimization budgets and checkpoint-selection protocol.

## Data and outputs

`A-proceeding/`, `SPP14/`, and `SPP1/` contain matching datasets, splits, and reference inputs. Collocation and single-instance arrays are included beside their runners. `SHA256SUMS.json` records release input checksums. Original recorded metadata and tables live under each study's `reference_results/`; fresh runs write to `runs/`.

For SPP fields, run `reconstruct_fields.m` in MATLAB after training, followed by `plot_fields.py`. The supplied FEM helpers reconstruct the vector field before plotting. Case 1 plotting uses the original triangular meshes. Figures share color limits across compared methods.

## Contributors

- [2577243392](https://github.com/2577243392) (Lida_Liu)
