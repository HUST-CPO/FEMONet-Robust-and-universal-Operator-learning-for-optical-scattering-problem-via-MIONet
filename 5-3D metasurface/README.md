# Case 5 — 3D Metasurface

Doubly-periodic **3D** metasurface with tetrahedral FEM and periodic boundary conditions (PBC). Predicts fields and reflection/transmission spectra.

## Dataset

| File | Status | Description |
|------|--------|-------------|
| `idx_3Dcase3_261.mat` | Included | 261-sample split |
| `deepOnet_data_3Dcase3_261.mat` | **Excluded from Git** (~851 MB) | Regenerate via `data_read.m` + `idx_3Dcase3_261.mat` |
| `E_*_pred_size_261_ddp_fft.mat` | Included | FFT-branch predictions |
| `Train_data_3Dcase3_*.mat` | **Omitted** | COMSOL raw (521/285/261) |
| `doublePBC_mesh.mat` | **Omitted** | Exported mesh |

See `samples and post proceeding/MISSING_MAT_FILES.txt`.

## Directory structure

```
5-3D metasurface/
├── NN/
│   ├── getdata.py
│   ├── cnn_branch_test1_DDP.py
│   ├── cnn_branch_test1_DDP_v4_261_norm.py  # Main training (V4)
│   ├── solver_time.py
│   └── solver_time_bicgstab.py
└── samples and post proceeding/
    ├── main_case1.m              # COMSOL 3D sampling
    ├── getMesh.m                 # Mesh export
    ├── data_read.m               # 285-sample preprocessing
    ├── test_single.m             # Single-sample |E| slices
    ├── PlotR.m                   # R–T spectra comparison
    ├── assembly_*.m, get_*.m     # 3D FEM helpers
    ├── mat_data/
    └── comsol_project/           # doublePBC.mph
```

## Workflow (261 main pipeline)

1. `main_case1.m` → `Train_data_3Dcase3_261.mat` (or via 521→285→261 subset chain)
2. Preprocess with `data_read.m` if rebuilding `deepOnet` (261 set may already exist)
3. `torchrun NN/cnn_branch_test1_DDP_v4_261_norm.py` → `E_*_pred_size_261_ddp.mat`
4. `test_single.m` / `PlotR.m` → `pdf_results/`

## COMSOL model

`doublePBC.mph` in `comsol_project/`.

## Notes

- Permittivity grids are 128³; branch ingests ε and wavelength λ (MLP, no FFT in V4).
- `PlotR.m` compares predicted vs. reference reflection/transmission using Poynting flux integrals.
- Python loaders must handle MATLAB v7.3 via `h5py` (implemented in `cnn_branch_test1_DDP.py`).
