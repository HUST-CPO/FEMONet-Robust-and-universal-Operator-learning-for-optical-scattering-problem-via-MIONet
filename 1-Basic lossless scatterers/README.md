# Case 1 — Basic Lossless Scatterers

2D transverse-magnetic (TM) scattering from lossless dielectric objects. COMSOL generates sparse FEM systems; a **CNN-DeepONet** with FEM residual loss is the main method (`NN/cnn_branch_test2.py`).

**Ablation baselines** (VPINN / Modified VPINN / VPINN-DeepONet): see [`ABLATION_README.md`](ABLATION_README.md).

## Dataset

| File | Samples | Description |
|------|---------|-------------|
| `Train_data_A_1558.mat` | 1558 | Main training set (models A1–A5) |
| `Train_data_A_unseen.mat` | 320 | Generalization set (models A6–A10) |
| `deepOnet_data_A_*.mat` | — | Network-ready tensors |
| `idx_A_*.mat` | — | Train/test split indices |
| `E_*_pred_size_*.mat` | — | Main method predictions |

**Status:** all `mat_data` files are included. See `samples and post proceeding/MISSING_MAT_FILES.txt`.

## Directory structure

```
1-Basic lossless scatterers/
├── ABLATION_README.md          # VPINN comparison overview
├── NN/                         # Main FEMO/MIONet training
├── VPINN single acse/          # Single-mesh VPINN (split BC loss)
├── Modified VPINN single acse/ # Single-mesh unified weak form
├── VPINN-DeepOnet/             # 1558-sample DeepONet + VPINN
└── samples and post proceeding/
    ├── Main_Scattering*.m, data_read_*.m, outputE*.m
    ├── mat_data/
    ├── comsol_project/
    ├── pdf_results/
    └── pdf_results_unseen/
```

## Main workflow (1558)

1. `Main_Scattering.m` → `Train_data_A_1558.mat`
2. `data_read_Ez_1558.m` → `deepOnet_data_A_1558.mat`, `idx_A_1558.mat`
3. `python NN/cnn_branch_test2.py` → `E_*_pred_size_1558.mat`
4. `outputE.m` → `pdf_results/`

### Unseen generalization

1. `Main_Scattering_unseen.m` → `Train_data_A_unseen.mat`
2. `data_read_Ez.m` → `deepOnet_data_A_unseen.mat`
3. Train NN → `E_*_pred_size_320_unseen.mat`
4. `outputE_unseen.m` → `pdf_results_unseen/`

## Ablation quick start

```bash
cd "VPINN single acse" && python "SBC vpinn.py"
cd "../Modified VPINN single acse" && python "SBC weak form.py"
cd "../VPINN-DeepOnet" && python deeponet_vpinn_5.py
```

Details: [VPINN single case](VPINN%20single%20acse/README_SBC_vpinn.md) · [Modified VPINN](Modified%20VPINN%20single%20acse/README_SBC_weak_form.md) · [VPINN-DeepOnet](VPINN-DeepOnet/README.md)

## COMSOL models

Place `scatteringA1.mph` … `scatteringA10.mph` in `samples and post proceeding/comsol_project/`.

## Notes

- Case 1/2/3 sampling scripts call `GetEdge()`; copy `mesh/GetEdge.m` from Case 4 or s1 if needed.
- Set `CUDA_VISIBLE_DEVICES` or `PINN_CUDA_DEVICE` in Python scripts for GPU selection.
