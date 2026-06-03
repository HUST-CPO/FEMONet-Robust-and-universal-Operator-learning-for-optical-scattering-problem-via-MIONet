# VPINN-DeepOnet — DeepONet + unified weak-form VPINN

Self-contained script [`deeponet_vpinn_5.py`](deeponet_vpinn_5.py): **CNN-DeepONet** branch (permittivity ε) + **trunk MLP** (coordinates), trained on **1558** scattering samples with a **single unified Robin weak-form** physics loss (same formulation as [`Modified VPINN single acse/SBC weak form.py`](../Modified%20VPINN%20single%20acse/SBC%20weak%20form.py) — no separate `loss_bc`).

## Requirements

- Python 3.10+, PyTorch, NumPy, SciPy, Matplotlib
- GPU recommended (`PINN_CUDA_DEVICE=0` by default)

```bash
pip install -r ../../requirements-cpu.txt   # or requirements-gpu.txt on cluster
```

## Input data

| File | Description | In Git? |
|------|-------------|---------|
| `Train_data_A1_all.mat` | Mesh + `Esz_train/test`, domain IDs (~90 MB) | Included in this folder |
| `Eplison_train/test` | Inside same `.mat` — ε maps for CNN branch | same file |

Alternative: point `scatter_mat` / `deep_mat` in `main()` to  
`samples and post proceeding/mat_data/Train_data_A_1558.mat` if the local copy is removed.

## Run

```bash
cd "1-Basic lossless scatterers/VPINN-DeepOnet"
export PINN_CUDA_DEVICE=0   # optional
python deeponet_vpinn_5.py
```

Outputs go to `compare_1558prj_mse/` (created at runtime). Committed snapshots may also exist in this folder root.

## Outputs

| File | Content |
|------|---------|
| `deeponet_vpinn_5_model.pth` | Trained weights |
| `deeponet_vpinn_5_history.csv` / `.npz` | Loss and MSE history |
| `deeponet_vpinn_5_training_curves.png` | Weak-form loss |
| `deeponet_vpinn_5_mse_train_test.png` | Train/test MSE vs epoch |
| `E_train_pred_deeponet_vpinn_5.mat` | Per-sample Ez predictions (train) |
| `E_test_pred_deeponet_vpinn_5.mat` | Per-sample Ez predictions (test) |
| `train_test_sample_index_deeponet_vpinn_5.mat` | `trainIdx`, `testIdx` |
| `all_fields_train/test_*/*.png` | \|Ez\| true vs pred per sample |

## Loss formulation

Unified weak form for scattered field `u_s`:

- Volume: `∫ ∇u·∇φ − ∫ k₀² n² u φ`
- Boundary Robin: `∫_Γ i k₀ u φ` (assembled on outer edges)
- RHS: `∫ f φ`, with `f = k₀² (n² − n_bg²) E_inc`

Normalized MSE of nodal residuals; train/test MSE tracks complex modulus vs COMSOL `Esz_*` labels.

## Hyperparameters (default in `main()`)

| Parameter | Value |
|-----------|-------|
| `hidden_channel` | 128 |
| `output_dim` | 128 |
| `batch_size` | 32 |
| `epochs` | 1000 |
| `lr` | 1e-3 |
| λ | 1.55 µm |

## Link to paper figures

Copy prediction `.mat` files to `samples and post proceeding/mat_data/` and run the plotting module for VPINN MSE overlay — see [`ABLATION_README.md`](../ABLATION_README.md) and [`plot loss …/MISSING_MAT_FILES.txt`](../../plot%20loss%20and%20mse%20histograms%20and%20guassion%20fits/MISSING_MAT_FILES.txt).

## Version note

`deeponet_vpinn_5` uses the **unified weak form** (v3+). Earlier variants (`deeponet_vpinn_2` with split `loss_bc`) are superseded by this script.
