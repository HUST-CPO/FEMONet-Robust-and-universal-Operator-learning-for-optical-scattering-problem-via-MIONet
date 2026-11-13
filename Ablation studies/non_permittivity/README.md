# Permittivity-input reference cases

Three additional reference configurations retain their supplied networks and real-valued FEM energy objectives. Each directory contains its matching prepared data and a training entry point. Case 1 uses A1_847, Case 2 uses B1_847, and Case 3 uses C1_3388. Run `python assemble_data.py` here before Case 3, then follow each case README. MAT compression preserves the stored arrays.

## Objective

These configurations predict a real field and optimize `0.5 * u.T @ real(A) @ u - u.T @ real(b)`. The data loader retains the supplied complex arrays; the energy evaluation uses their real components. Case 1 and Case 2 each use 678 training / 169 test samples; Case 3 uses 2710 / 678.
