"""Check release inputs, complex residuals, derivatives, and sampling on CPU."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def check_module(folder, statement):
    subprocess.run([sys.executable, "-c", statement], cwd=ROOT / folder, check=True)


def main():
    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for name, expected in json.loads((ROOT / "SHA256SUMS.json").read_text()).items():
        digest = hashlib.sha256()
        with (ROOT / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == expected, name
    for folder in (
        "ablation_no_ilu_case1",
        "ablation_no_ilu_case4",
        "ablation_no_ilu_s2",
    ):
        check_module(
            folder,
            "from run_ablation import check_raw_gradient; print(check_raw_gradient())",
        )
    check_module(
        "ablation_pinn_case1",
        "from run_pinn import derivative_check; print(derivative_check())",
    )
    check_module(
        "ablation_label_efficiency_case1_epochs",
        "from run_full_dataset import check; check()",
    )

    import numpy as np
    from scipy.io import loadmat

    data = loadmat(ROOT / "A-proceeding/deepOnet_data_A1_all.mat")
    split = loadmat(ROOT / "A-proceeding/idx_A1_all.mat")
    train, test = split["trainIdx"].ravel(), split["testIdx"].ravel()
    assert len(train) == 1246 and len(test) == 312
    assert not set(train) & set(test)
    pred = loadmat(ROOT / "A-proceeding/E_test_pred_size_ALL.mat")["E_pred"]
    lengths = data["coord_len_test"].ravel()
    mask = np.arange(pred.shape[1])[None, :] < lengths[:, None]
    mse = np.sum(np.abs(pred - data["Ez_test"])[mask] ** 2) / lengths.sum()
    np.testing.assert_allclose(mse, 4.390900391795585e-5, rtol=1e-10)
    print(f"PASS: release inputs, CPU checks, split, and archived MSE {mse:.15g}")


if __name__ == "__main__":
    main()
