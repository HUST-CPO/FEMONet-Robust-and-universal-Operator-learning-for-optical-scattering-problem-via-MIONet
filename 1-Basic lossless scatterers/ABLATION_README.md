# Ablation baselines — Case 1 (lossless scatterers)

Comparison experiments against the main **FEM-residual CNN-DeepONet** pipeline (`NN/cnn_branch_test2.py`). All three baselines solve the same 2D TM Robin scattering problem on a triangular mesh; they differ in **network architecture** and **how the Robin boundary is enforced in the loss**.

## Methods

| Folder | Method | Network | Boundary in loss |
|--------|--------|---------|------------------|
| [`VPINN single acse/`](VPINN%20single%20acse/README_SBC_vpinn.md) | **VPINN (split BC)** | MLP `(x,y) → (u_r, u_i)` | `L = L_var + w_bc·L_bc` — volume weak form + separate Robin BC loss |
| [`Modified VPINN single acse/`](Modified%20VPINN%20single%20acse/README_SBC_weak_form.md) | **Modified VPINN (unified weak form)** | Same MLP | Single normalized weak-form residual (volume + Robin boundary assembled together) |
| [`VPINN-DeepOnet/`](VPINN-DeepOnet/README.md) | **DeepONet + VPINN** | CNN branch (ε) + trunk MLP; multi-sample | Same unified weak form as Modified VPINN; trained on 1558 COMSOL samples |

Main method (this repo): sparse FEM residual on edge-element DOFs — see [`NN/cnn_branch_test2.py`](NN/cnn_branch_test2.py).

## Single-scatterer setup (VPINN / Modified VPINN)

- One fixed mesh: `MeshData_Robin.mat` (circular scatterer, Robin outer BC)
- Plane-wave incidence: `E_inc = exp(-i k0 x)`
- Default: λ = 1.55 µm, `n_bg = 1`, `n_scatter = 1.45`, `r_scatter = 0.2` µm

## Multi-sample setup (VPINN-DeepOnet)

- Mesh + labels from `Train_data_A1_all.mat` (1558 samples, same as Case 1 main set)
- Permittivity maps from `Eplison_train/test` in the same file
- Predictions: `E_train/test_pred_deeponet_vpinn_5.mat` — used in MSE histogram overlay (`error_MSE_A_with_vpinn.pdf`)

## Quick run

```bash
cd "1-Basic lossless scatterers"

# Single mesh — split BC VPINN
cd "VPINN single acse"
python "SBC vpinn.py"

# Single mesh — unified weak form
cd "../Modified VPINN single acse"
python "SBC weak form.py"

# 1558 samples — DeepONet + VPINN
cd "../VPINN-DeepOnet"
python deeponet_vpinn_5.py
```

Set `PINN_CUDA_DEVICE` (DeepONet script) or edit `device` in single-case scripts for GPU selection.

## Outputs vs main pipeline

| Output | VPINN single | Modified VPINN | VPINN-DeepOnet |
|--------|--------------|----------------|----------------|
| Loss curves | `result/SBC_vpinn_loss_curve.png` | `result/SBC_weak_loss_curve.png` | `deeponet_vpinn_5_training_curves.png` |
| \|E\| field | `result/SBC_vpinn_normE.png` | `result/SBC_weak_form_normE.png` | `all_fields_*/*.png` |
| Predictions `.mat` | `E_pred_vpinn.mat` | `E_pred_weak.mat` | `E_*_pred_deeponet_vpinn_5.mat` |

Copy DeepONet prediction `.mat` files to `samples and post proceeding/mat_data/` if running [`plot loss …/compute_mse_cache`](../plot%20loss%20and%20mse%20histograms%20and%20guassion%20fits/README.md) VPINN overlay.

## Large data note

`VPINN-DeepOnet/Train_data_A1_all.mat` (~90 MB) duplicates Case 1 training data. If omitted from a clone, regenerate with `Main_Scattering.m` or copy from `samples and post proceeding/mat_data/Train_data_A_1558.mat` after renaming fields (see `VPINN-DeepOnet/README.md`).
