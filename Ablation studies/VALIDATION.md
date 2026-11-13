# Release validation

Validated on 2026-11-13 with Python 3.11 and PyTorch 2.5.1 CPU.

- All packaged Python files parse; all nine ablation training entry points accept `--help`.
- Computational ASTs of the archived network and loader snapshots match the local experiment sources after excluding comments and docstrings.
- Complex sparse residual values and gradients agree with dense algebra for all three ILU cases.
- PINN polynomial Laplacians, parameter gradients, Robin signs, and source signs pass.
- Full-subset label sampling preserves nested subsets, matched DD/Hybrid ordering, 1000 epochs, and 39000 updates.
- The Case 1 split has 1246 training and 312 test samples with disjoint indices.
- Archived Case 1 predictions reproduce test MSE 4.390900391795585e-5.
- The Case 1 checkpoint loads strictly and reproduces the first two saved test predictions within rtol 1e-9 and atol 1e-10.
- Bundled input checksums and Case 3 data reassembly pass.

The release validation covers packaging and CPU numerical checks. Full training is run using the CUDA commands in each study README. Recorded experiment tables are included in `reference_results/`.
