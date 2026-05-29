# Case 3 — Multiple Metallic Scatterers

2D TM scattering from **multiple metallic scatterers** (merged COMSOL models C1 + C2). 3,456 samples with reproducible 80/20 train/test split.

## Dataset

| File | Status | Description |
|------|--------|-------------|
| `idx_C_3456.mat` | Included | Split (seed 20260407) |
| `deepOnet_data_C_3456.mat` | Included | Preprocessed tensors |
| `Train_data_C_3456.mat` | **Omitted** | COMSOL raw data |
| `E_*_pred_size_3456.mat` | **Omitted** | NN predictions |

See `samples and post proceeding/MISSING_MAT_FILES.txt`.

## Directory structure

```
3-Multiple metallic scatterers/
├── NN/
│   ├── getdata.py
│   ├── cnn_branch_test_1.py       # Main training script
│   └── solver_time_bicgstab.py
└── samples and post proceeding/
    ├── Main_Scattering_C_cnn.m    # C1+C2 merged sampling
    ├── data_read_Ez_C_3456.m
    ├── outputE_v2.m
    ├── mat_data/
    └── comsol_project/            # scatteringC1.mph, scatteringC2.mph
```

## Workflow

1. `Main_Scattering_C_cnn.m` → `Train_data_C_3456.mat`
2. `data_read_Ez_C_3456.m` → `deepOnet_data_C_3456.mat` (skip if already present)
3. `python NN/cnn_branch_test_1.py` → `E_*_pred_size_3456.mat`
4. `outputE_v2.m` → `pdf_results/`

## COMSOL models

`scatteringC1.mph`, `scatteringC2.mph` in `comsol_project/`.

## Notes

- `outputE_v2.m` visualizes both `Ez` and `Ebz` channels where applicable.
- `deepOnet_data` is stored in MATLAB v7.3 (HDF5); Python loaders should use `h5py` or SciPy with v7.3 support.
