# Modified VPINN — single scatterer (unified weak form)

[`SBC weak form.py`](SBC%20weak%20form.py): **VPINN** with Robin conditions embedded in a **single unified weak-form residual** — no separate `loss_bc`. This formulation is reused in [VPINN-DeepOnet](../VPINN-DeepOnet/deeponet_vpinn_5.py) for multi-sample training.

Compare with [split-BC VPINN](../VPINN%20single%20acse/README_SBC_vpinn.md). Overview: [ABLATION_README.md](../ABLATION_README.md).

## Run

```bash
cd "1-Basic lossless scatterers/Modified VPINN single acse"
python "SBC weak form.py"
```

Requires **`MeshData_Robin.mat`** (or `MeshData_new.mat`).

Default: 15 000 epochs, Adam `lr=1e-3`, GPU `cuda:0` if available.

## Weak form (scattered field u_s)

```
∫_Ω ∇u_s·∇v dΩ − ∫_Ω k₀² n² u_s v dΩ − ∫_Γ i k₀ u_s v dΓ = ∫_Ω f v dΩ
```

- Test functions v: piecewise linear hat functions on the mesh
- Γ: outer boundary (Robin, homogeneous)
- `f = k₀² (n² - n_bg²) E_inc`, `E_inc = exp(-i k₀ x)`

Loss: normalized mean square of nodal residuals `(lhs - f)`, divided by `mean(f²)`.

## Loss

Single term: **`vpinn_weak_loss`** — volume gradient + mass + boundary Robin terms assembled to global nodes, then MSE normalized by RHS energy.

## Outputs

| File | Description |
|------|-------------|
| `result/SBC_weak_loss_curve.png` | Weak-form loss vs epoch |
| `result/SBC_weak_form_normE.png` | \|E_total\| on a regular grid |
| `E_pred_weak.mat` | 128×128 \|E_total\| grid |
| `SBC_weak_form_model.pth` | Weights |
| `SBC_weak_form_history.npz` | History |

## Default physical parameters

Same as VPINN single case: λ = 1.55 µm, `n_bg = 1`, `n_scatter = 1.45`, `r_scatter = 0.2` µm, MLP 6×128.

## Relation to DeepONet baseline

| Item | This script | `deeponet_vpinn_5.py` |
|------|-------------|------------------------|
| Network | MLP on (x,y) | CNN(ε) + trunk(x,y) |
| Samples | 1 mesh | 1558 meshes |
| Weak form | Unified | Same unified form |
