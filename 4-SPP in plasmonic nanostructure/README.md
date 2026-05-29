# Case 4 — SPP in Plasmonic Nanostructure (SPP14)

2D surface-plasmon-polariton (SPP) problem with **in-house edge-element FEM** assembly (Nédélec-type basis) coupled to COMSOL mesh import. 41 wavelength samples (λ = 1.0–1.4).

## Dataset

All files in `mat_data/` are included:

- `Train_data_SPP14_41.mat`
- `deepOnet_data_SPP14_41.mat`
- `idx_SPP14_41.mat`
- `E_train_pred_size_41.mat`, `E_test_pred_size_41.mat`

See `samples and post proceeding/MISSING_MAT_FILES.txt`.

## Directory structure

```
4-SPP in plasmonic nanostructure/
├── NN/
│   ├── getdata.py
│   ├── cnn_branch_test1.py        # Main training script
│   └── solver_time_bicgstab.py
└── samples and post proceeding/
    ├── Main_SPP14.m               # COMSOL + FEM assembly
    ├── data_read_Ez.m
    ├── outputE_v3.m               # |E| field comparison PDFs
    ├── function/                  # Basis functions (BF_Et, BF_Ez, …)
    ├── kernel/                    # FEM assembly (Equ, Inc, Out, Out2–4)
    ├── mesh/                      # Edge DOF utilities
    ├── post/                      # GetExEy, PlotnormE, …
    ├── mat_data/
    └── comsol_project/            # case14_2D_freq.mph
```

## Workflow

1. `Main_SPP14.m` → `Train_data_SPP14_41.mat`
2. `data_read_Ez.m` → `deepOnet_data_SPP14_41.mat`, `idx_SPP14_41.mat`
3. `python NN/cnn_branch_test1.py` → `E_*_pred_size_41.mat`
4. `outputE_v3.m` → `pdf_results/`

## FEM helpers

Case 4 uses custom boundary assemblies `AssemblyOfOut3` and `AssemblyOfOut4` in `kernel/` (not present in s1). Scripts add `function/`, `kernel/`, `mesh/`, and `post/` to the MATLAB path automatically.

## COMSOL model

`case14_2D_freq.mph` in `comsol_project/`.

## Notes

- Post-processing reconstructs Cartesian field components via `GetExEy` before plotting |E|.
- CNN branch uses InstanceNorm and coordinate channels; see `cnn_branch_test1.py` for hyperparameters.
