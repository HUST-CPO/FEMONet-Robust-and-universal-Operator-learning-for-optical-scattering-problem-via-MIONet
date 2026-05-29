# Case s1 — SPP in Plasmonic Nanostructure (SPP1)

2D SPP benchmark with periodic boundaries and a **projection matrix P** mapping edge DOFs. 71 geometry-factor samples (factor = 0.2–0.9). Shares FEM helper layout with Case 4.

## Dataset

All files in `mat_data/` are included:

- `Train_data_SPP1_71.mat`
- `deepOnet_data_SPP1_71.mat`
- `idx_SPP1_71.mat`
- `E_train_pred_size_71.mat`, `E_test_pred_size_71.mat`

See `samples and post proceeding/MISSING_MAT_FILES.txt`.

## Directory structure

```
s1-SPP in plasmonic nanostructure/
├── NN/
│   ├── getdata.py
│   └── cnn_branch_test1.py       # Main training script
└── samples and post proceeding/
    ├── Main_SPP1.m
    ├── data_read.m
    ├── outputE_v3.m
    ├── function/ kernel/ mesh/ post/
    ├── mat_data/
    └── comsol_project/           # case1_2D.mph
```

## Workflow

1. `Main_SPP1.m` → `Train_data_SPP1_71.mat`
2. `data_read.m` → `deepOnet_data_SPP1_71.mat`, `idx_SPP1_71.mat`
3. `python NN/cnn_branch_test1.py` → `E_*_pred_size_71.mat`
4. `outputE_v3.m` → `pdf_results/`

## Differences from Case 4

| Item | Case 4 (SPP14) | s1 (SPP1) |
|------|----------------|-----------|
| Samples | 41 (λ sweep) | 71 (geometry factor) |
| PBC | Standard boundaries | `findTris`, `GetcopyOfBedge`, matrix `P` |
| `kernel/` | Includes `AssemblyOfOut3/4` | Five assembly files only |
| `data_read` | `data_read_Ez.m` | `data_read.m` (uses `Matrix{8}` = P) |

## COMSOL model

`case1_2D.mph` in `comsol_project/`.
