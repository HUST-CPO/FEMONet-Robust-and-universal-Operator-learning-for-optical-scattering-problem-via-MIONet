# Loss Curves & MSE Histograms (Gaussian Fits)

MATLAB scripts for **paper figures**: training-loss curves, per-sample MSE histograms, and log-domain Gaussian fits. Also includes an ablation heatmap for Case 1 (model A).

## Directory layout

```
plot loss and mse histograms and guassion fits/
├── get_plot_paths.m          % path helper
├── plot_loss_curve.m         % generic loss-curve exporter
├── plot_loss_modelAV.m       % Case A vs ablation (modelAV)
├── compute_mse_cache.m       % per-sample MSE from mat_data
├── plot_mse_gaussian.m       % histogram + log-Gaussian fit
├── run_all_loss_curves.m     % batch: all loss logs
├── run_all_mse_histograms.m  % batch: MSE figures (ready cases)
├── MISSING_MAT_FILES.txt     % omitted large .mat + how to generate
├── loss_log/                 % INPUT: training logs
├── mse_cache/                % INTERMEDIATE: per-sample MSE .mat
├── plot_results/             % OUTPUT: PDF / SVG figures
└── ablation/
    ├── ablation_grid_progress.csv
    └── plot_heatmap.m
```

## Requirements

- MATLAB R2019b+ (uses `exportgraphics`)
- Statistics Toolbox (`normcdf`, `ksdensity` for KDE script)
- Case `.mat` files under each case’s `samples and post proceeding/mat_data/`

## Quick start

```matlab
cd('plot loss and mse histograms and guassion fits')

% 1) All training-loss curves -> plot_results/
run_all_loss_curves

% 2) MSE histograms (cases with complete data) -> plot_results/
run_all_mse_histograms

% 3) Ablation heatmap -> plot_results/
cd ablation
plot_heatmap
cd ..
```

## Inputs

### Loss curves (`loss_log/`)

| File | Case | Log format |
|------|------|------------|
| `modelA_loss.txt` | 1 — Basic scatterers | `Epoch N, Total Loss: …, test Loss …` |
| `modelB_loss.log` | 2 — Single metallic | same |
| `modelC_loss.txt` | 3 — Multiple metallic | `epoch fem_loss test_loss lr` columns |
| `model3D_loss.log` | 5 — 3D metasurface | Epoch format |
| `modelSPP1_loss.txt` | s1 — SPP1 | Epoch format |
| `modelSPP14_loss.txt` | 4 — SPP14 | Epoch format |
| `modelAV_loss.csv` | Case 1 ablation | `epoch,mse_train,mse_test` |

Copy new training logs from SLURM / local runs into `loss_log/` before plotting.

### MSE histograms (case `mat_data/`)

`compute_mse_cache(case_id)` reads predictions and ground truth from:

| `case_id` | Case folder | Key files |
|-----------|-------------|-----------|
| `A` | `1-Basic lossless scatterers` | `Train_data_A_1558`, `E_*_pred_size_1558`, `idx_A_1558` |
| `SPP14` | `4-SPP …` | `Train_data_SPP14_41`, `E_*_pred_size_41` |
| `SPP1` | `s1-SPP …` | `Train_data_SPP1_71`, `E_*_pred_size_71` |
| `3D` | `5-3D metasurface` | `Train_data_3Dcase3_261`, `E_*_261_ddp_fft` (field scale ×1/10) |
| `C` | `3-Multiple metallic` | needs `Train_data_C_3456`, `E_*_3456` (often omitted) |
| `B` | `2-Single metallic` | needs `Train_data_B1_50688`, `E_*_50688` (often omitted) |

**Repository snapshot:** `A`, `SPP14`, `SPP1` have complete `mat_data`; `3D`/`C`/`B` need large files omitted from Git — see [`MISSING_MAT_FILES.txt`](MISSING_MAT_FILES.txt) and each case’s `MISSING_MAT_FILES.txt` (reason: **file size**, plus generation steps).

## Outputs (`plot_results/`)

| Figure | Script |
|--------|--------|
| `modelA_loss.pdf/.svg` | `run_all_loss_curves` |
| `modelB_loss`, `modelC_loss`, `model3D_loss`, `modelSPP*_loss` | same |
| `modelA_modelAV_loss` | ablation comparison |
| `error_MSE_A.pdf`, `error_MSE_SPP14.pdf`, … | `run_all_mse_histograms` |
| `best_test_fem_loss_heatmap.pdf` | `ablation/plot_heatmap.m` |

Intermediate caches: `mse_cache/<case_id>_train.mat`, `_test.mat` (variables `sample_mse`, `mse_global`).

## Per-case workflow

```matlab
% Example: Case 4 (SPP14)
compute_mse_cache('SPP14');
plot_mse_gaussian('SPP14', [-6 0], 'error_MSE_SPP14');

% Example: single loss curve
P = get_plot_paths();
plot_loss_curve(fullfile(P.log_dir, 'modelSPP14_loss.txt'), 'modelSPP14_loss', 'epoch');
```

## Optional VPINN overlay (Case A)

To reproduce `error_MSE_A_with_vpinn.pdf`, place VPINN prediction `.mat` files and re-run a custom comparison (not automated in `run_all_mse_histograms`). Requires `E_test_pred_deeponet_vpinn_5.mat` and matching `Esz_*` fields in `deepOnet_data_A_1558.mat`.

## Adding a new case

1. Export training log to `loss_log/`.
2. Ensure `mat_data/` contains `Train_data_*`, `deepOnet_*`, `idx_*`, `E_train/test_pred_*`.
3. Add a branch in `compute_mse_cache.m` → `case_config`.
4. Run `compute_mse_cache('NEWID')` and `plot_mse_gaussian('NEWID', …)`.
