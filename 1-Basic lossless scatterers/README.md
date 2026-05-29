# Case 1 — Basic Lossless Scatterers

2D transverse-magnetic (TM) scattering from lossless dielectric objects. COMSOL generates sparse FEM systems; a CNN-DeepONet predicts the electric field with a physics residual loss.

## Dataset

| File | Samples | Description |
|------|---------|-------------|
| `Train_data_A_1558.mat` | 1558 | Main training set (models A1–A5) |
| `Train_data_A_unseen.mat` | 320 | Generalization set (models A6–A10) |
| `deepOnet_data_A_*.mat` | — | Network-ready tensors |
| `idx_A_*.mat` | — | Train/test split indices |
| `E_*_pred_size_*.mat` | — | NN predictions |

**Status:** all `mat_data` files are included. See `samples and post proceeding/MISSING_MAT_FILES.txt`.

## Directory structure

```
1-Basic lossless scatterers/
├── NN/
│   ├── getdata.py              # PyTorch Dataset
│   ├── cnn_branch_test2.py     # Main training script
│   └── solver_time_bicgstab.py # Iterative solver baseline
└── samples and post proceeding/
    ├── Main_Scattering.m           # COMSOL sampling (1558)
    ├── Main_Scattering_unseen.m    # COMSOL sampling (unseen)
    ├── data_read_Ez_1558.m         # Preprocess 1558 set
    ├── data_read_Ez.m              # Preprocess unseen set
    ├── outputE.m                   # Figures for 1558 set
    ├── outputE_unseen.m            # Figures for unseen set
    ├── mat_data/
    ├── comsol_project/             # scatteringA1.mph – A10.mph
    ├── pdf_results/
    └── pdf_results_unseen/
```

## Workflow

### 1558 dataset

1. `Main_Scattering.m` → `Train_data_A_1558.mat`
2. `data_read_Ez_1558.m` → `deepOnet_data_A_1558.mat`, `idx_A_1558.mat`
3. `python NN/cnn_branch_test2.py` → `E_*_pred_size_1558.mat`
4. `outputE.m` → `pdf_results/`

### Unseen generalization

1. `Main_Scattering_unseen.m` → `Train_data_A_unseen.mat`
2. `data_read_Ez.m` → `deepOnet_data_A_unseen.mat`
3. Train with the corresponding NN script → `E_*_pred_size_320_unseen.mat`
4. `outputE_unseen.m` → `pdf_results_unseen/`

## COMSOL models

Place `scatteringA1.mph` … `scatteringA10.mph` in `samples and post proceeding/comsol_project/`.

## Notes

- Case 1/2/3 sampling scripts call `GetEdge()`; copy `mesh/GetEdge.m` from Case 4 or s1 if needed.
- Set `CUDA_VISIBLE_DEVICES` in the Python scripts for GPU selection.
