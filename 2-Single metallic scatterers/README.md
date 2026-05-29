# Case 2 — Single Metallic Scatterers

2D TM scattering from **single metallic** inclusions with lossy permittivity. Larger dataset (50,688 samples) trained with multi-GPU DDP.

## Dataset

| File | Status | Description |
|------|--------|-------------|
| `idx_B1_50688.mat` | Included | Fixed train/test split |
| `Train_data_B1_50688.mat` | **Omitted** (large) | COMSOL raw samples |
| `deepOnet_data_B1_50688.mat` | **Omitted** | Preprocessed tensors |
| `E_*_pred_50688*.mat` | **Omitted** | NN predictions |

See `samples and post proceeding/MISSING_MAT_FILES.txt` for the full list and reproduction order.

## Directory structure

```
2-Single metallic scatterers/
├── NN/
│   ├── getdata.py
│   ├── cnn_branch_test.py           # Single-GPU training
│   ├── cnn_branch_test1_DDP.py        # Main DDP training
│   └── cnn_branch_test1_DDP_fft.py  # DDP + FFT branch variant
└── samples and post proceeding/
    ├── Main_Scattering.m              # COMSOL B1–B7 sampling
    ├── data_read_Ez.m                 # 1152 subset preprocessing
    ├── data_read_Ez_B1_50688.m        # 50688 set (uses idx_B1_50688)
    ├── outputE_v2.m
    ├── mat_data/
    └── comsol_project/                # scatteringB1.mph – B7.mph
```

## Workflow (50,688 main pipeline)

1. `Main_Scattering.m` → `Train_data_B1_50688.mat`
2. `data_read_Ez_B1_50688.m` (with `idx_B1_50688.mat`) → `deepOnet_data_B1_50688.mat`
3. `torchrun --nproc_per_node=N NN/cnn_branch_test1_DDP.py` → `E_*_pred_size_50688_ddp.mat`
4. `outputE_v2.m` → `pdf_results/`

## COMSOL models

`scatteringB1.mph` … `scatteringB7.mph` in `comsol_project/`.

## Notes

- Branch network may include incident-field (`Ebz`) and wavelength channels.
- `cnn_branch_test1_DDP_fft.py` replaces part of the branch with an FFT-based MLP.
