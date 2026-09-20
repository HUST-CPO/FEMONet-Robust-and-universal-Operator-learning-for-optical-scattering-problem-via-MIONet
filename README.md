# FEMONet: Robust and Universal Operator Learning for Optical Scattering via MIONet

Code release accompanying the paper: **Learning light scattering from operator parameter spaces to Galerkin-consistent solution spaces** for fast prediction of electromagnetic fields in nanophotonic and metasurface problems.

This repository contains six self-contained case studies. Each case follows the same high-level pipeline:

```
COMSOL parameter sweep  →  Train_data_*.mat
        ↓
MATLAB preprocessing    →  deepOnet_data_*.mat + idx_*.mat
        ↓
PyTorch training        →  E_*_pred_*.mat
        ↓
MATLAB post-processing  →  pdf_results/
```

## Repository layout


| Folder                                                                                 | Problem                              | Samples       | Notes                                      |
| -------------------------------------------------------------------------------------- | ------------------------------------ | ------------- | ------------------------------------------ |
| [1-Basic lossless scatterers](1-Basic%20lossless%20scatterers/README.md)               | 2D lossless dielectric scatterers    | 1558 + unseen | Full dataset; includes VPINN ablation baselines |
| [2-Single metallic scatterers](2-Single%20metallic%20scatterers/README.md)             | 2D single metallic scatterers        | 50688         | Large `.mat` files omitted; `idx` provided |
| [3-Multiple metallic scatterers](3-Multiple%20metallic%20scatterers/README.md)         | 2D multi-scatterer arrays            | 3456          | `idx` on Git; `deepOnet` ~1 GB excluded    |
| [4-SPP in plasmonic nanostructure](4-SPP%20in%20plasmonic%20nanostructure/README.md)   | 2D surface-plasmon structure (SPP14) | 41            | Full dataset + FEM helpers                 |
| [5-3D metasurface](5-3D%20metasurface/README.md)                                       | 3D doubly-periodic metasurface       | 261           | `deepOnet` ~851 MB excluded from Git       |
| [s1-SPP in plasmonic nanostructure](s1-SPP%20in%20plasmonic%20nanostructure/README.md) | 2D SPP supplyment experiment (SPP1)              | 71            | Full dataset + FEM helpers                 |
| [Plotting (loss & MSE)](plot%20loss%20and%20mse%20histograms%20and%20guassion%20fits/README.md) | Loss curves, MSE histograms, ablation heatmap | — | MATLAB; reads `loss_log/` + case `mat_data/` |


Each case contains:

- `NN/` — PyTorch training and inference scripts
- `samples and post proceeding/` — COMSOL sampling, data preparation, and figure export
  - `mat_data/` — `.mat` datasets and predictions
  - `comsol_project/` — COMSOL `.mph` models (not always included)
  - `pdf_results/` — exported figures
  - `MISSING_MAT_FILES.txt` — list of omitted large files and reproduction steps

## Requirements

### MATLAB

- MATLAB R2019b or later (recommended)
- [COMSOL LiveLink for MATLAB](https://www.comsol.com/livelink-for-matlab) (`mphload`, `mphinterp`, `mphmatrix`, `mphxmeshinfo`, `mphmeshstats`)
- COMSOL models in `comsol_project/` (see per-case README)

### Python

Full dependency documentation: **[docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)**.

| Environment | File | PyTorch | Use |
|-------------|------|---------|-----|
| **CPU** (local `deepXDE`) | `requirements-cpu.txt` | 2.7.0 (CPU) | Development, smoke tests |
| **GPU** (SLURM cluster) | `requirements-gpu.txt` | 2.5.1 (CUDA 12.4) | Training / DDP |

All `NN/*.py` scripts need only: `torch`, `numpy`, `scipy`, `h5py`, `matplotlib`, `tqdm`, `pandas`.

**CPU (local)**

```bash
conda activate deepXDE
python -m pip install -r requirements-cpu.txt
# or: conda env create -f environment-cpu.yml
```

**GPU (SLURM cluster)** — install on a GPU node, then use the exported lock file:

```bash
module load cuda/12.4
conda activate pinn-gpu
python -m pip install -r requirements-gpu.txt
# or: conda env create -f environment-gpu.yml
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Re-export after cluster env changes: `python -m pip freeze > requirements-gpu.txt`

Multi-GPU training (Cases 2, 5):

```bash
torchrun --nproc_per_node=N NN/cnn_branch_test1_DDP.py
```

### Paper figures (loss & MSE)

MATLAB scripts in [`plot loss and mse histograms and guassion fits/`](plot%20loss%20and%20mse%20histograms%20and%20guassion%20fits/README.md) generate training-loss curves, MSE histograms with log-Gaussian fits, and the Case 1 ablation heatmap.

```matlab
cd('plot loss and mse histograms and guassion fits')
run_all_loss_curves      % loss_log/ -> plot_results/
run_all_mse_histograms   % case mat_data/ -> plot_results/
```

Inputs: `loss_log/*.txt|log|csv` and per-case `mat_data/` (see [`MISSING_MAT_FILES.txt`](plot%20loss%20and%20mse%20histograms%20and%20guassion%20fits/MISSING_MAT_FILES.txt) — large `.mat` files omitted due to size; generation steps listed per case).

## Quick start (Case 1 — complete dataset)

```matlab
% MATLAB — optional if Train_data already present
cd('1-Basic lossless scatterers/samples and post proceeding')
Main_Scattering          % COMSOL sampling
data_read_Ez_1558          % build deepOnet dataset
```

```bash
cd "1-Basic lossless scatterers/NN"
python cnn_branch_test2.py   # train; writes predictions to mat_data/
```

```matlab
outputE                    % export comparison figures to pdf_results/
```

## Data policy

Large files (`Train_data_*.mat`, `deepOnet_data_*.mat`, `E_pred*.mat`) may be omitted from the repository due to size (GitHub **100 MB per-file limit**). Each case ships `MISSING_MAT_FILES.txt` describing:

- which files are present or missing
- which script generates each file
- how to reproduce the pipeline using provided `idx_*.mat` splits

**Not on GitHub (in `.gitignore`):**

| File | Size | Regenerate |
|------|------|------------|
| `deepOnet_data_C_3456.mat` | ~1.05 GB | Case 3: `data_read_Ez_C_3456.m` |
| `deepOnet_data_3Dcase3_261.mat` | ~851 MB | Case 5: `data_read.m` + `idx_3Dcase3_261.mat` |

## License

[Specify license before public release, e.g. MIT or Apache-2.0]

## Citation

If you use this code, please cite the accompanying paper (citation to be added upon publication).
