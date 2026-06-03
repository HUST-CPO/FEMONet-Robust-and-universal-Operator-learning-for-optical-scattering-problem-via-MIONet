# VPINN — single scatterer (split boundary loss)

[`SBC vpinn.py`](SBC%20vpinn.py): **VPINN** on a fixed triangular mesh. Robin boundary conditions are enforced via a **separate boundary loss** `loss_bc`; the volume weak form `loss_var` does **not** include boundary integrals.

Ablation baseline for Case 1 — compare with [Modified VPINN](../Modified%20VPINN%20single%20acse/README_SBC_weak_form.md) and [VPINN-DeepOnet](../VPINN-DeepOnet/README.md). Overview: [ABLATION_README.md](../ABLATION_README.md).

## Run

```bash
cd "1-Basic lossless scatterers/VPINN single acse"
python "SBC vpinn.py"
```

Requires **`MeshData_Robin.mat`** (or fallback `MeshData_new.mat`) in this directory.

Default: 15 000 epochs, Adam `lr=1e-3`, `w_bc=1`, GPU `cuda:0` if available.

## Governing equation (scattered field u_s)

```
∇² u_s + k₀² n² u_s = -k₀² (n² - n_bg²) E_inc
```

Incident field: `E_inc = exp(-i k₀ x)` → real/imag: `(cos(k₀x), -sin(k₀x))`.

## Loss

Total: **`L = L_var + w_bc · L_bc`** (default `w_bc = 1`).

| Term | Meaning |
|------|---------|
| **`L_var`** | Volume weak-form residual on **interior nodes only** |
| **`L_bc`** | Robin BC on boundary quadrature points: `∂u/∂n - i k₀ u = 0` |

## Outputs

Figures are saved under **`result/`** when run from this folder (move or set paths in script if needed):

| File | Description |
|------|-------------|
| `result/SBC_vpinn_loss_curve.png` | total / var / bc |
| `result/SBC_vpinn_normE.png` | Total field magnitude \|E\| |
| `E_pred_vpinn.mat` | 128×128 grid of \|E_total\| (in script output dir) |
| `Robin_vpinn_model.pth` | Saved weights |
| `Robin_vpinn_history.npz` | Loss history |

## Default physical parameters

| Parameter | Value |
|-----------|-------|
| Wavelength λ | 1.55 µm |
| n_bg | 1.0 |
| n_scatter | 1.45 |
| r_scatter | 0.2 µm |
| MLP | 6 layers × 128 hidden |

## Notes

- Folder name uses historical spelling `acse` (= single case).
- Mesh coordinates are auto-scaled to µm if stored in metres.
