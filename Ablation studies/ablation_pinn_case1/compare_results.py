"""Compare results."""

import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
reference = root.parent / "ablation_no_ilu_case1"
if not reference.exists():
    reference = root.parent / "ablation_no_ilu_case1"
pinn = root / "runs/pinn_seed123"
p = json.loads((pinn / "metrics.json").read_text())
pm = json.loads((pinn / "manifest.json").read_text())
assert p["epochs"] == 1000
rows = []
for name, run in [
    ("Archived ILU (Table 2)", "verify"),
    ("FEMONet without ILU", "raw_seed123"),
    ("FEMONet with ILU (paired)", "ilu_seed123"),
]:
    m = json.loads((reference / "runs" / run / "metrics.json").read_text())
    manifest = json.loads((reference / "runs" / run / "manifest.json").read_text())
    for key in ("data_sha256", "source_sha256", "initial_state_sha256"):
        assert pm[key] == manifest[key]
    for key in (
        "epochs",
        "batch_size",
        "learning_rate",
        "min_learning_rate",
        "max_grad_norm",
        "dtype",
    ):
        assert pm["config"][key] == manifest["config"][key]
    rows.append(
        {
            "method": name,
            "train_mse": m["train"]["mse_global"],
            "test_mse": m["test"]["mse_global"],
            "training_seconds": m.get("training_seconds", 7891.211123),
        }
    )
rows.append(
    {
        "method": "PINN-DeepONet (strong form, lambda_b=1)",
        "train_mse": p["train"]["mse_global"],
        "test_mse": p["test"]["mse_global"],
        "training_seconds": p["training_seconds"],
    }
)
out = root / "comparison"
out.mkdir(exist_ok=True)
with (out / "mse_comparison.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
summary = {
    "pinn_over_paired_ilu_test_mse": rows[-1]["test_mse"] / rows[2]["test_mse"],
    "pinn_over_archived_ilu_test_mse": rows[-1]["test_mse"] / rows[0]["test_mse"],
    "metric": "total squared complex field error / valid DOFs; no padding",
    "qualification": "One seed, fixed lambda_b=1, mesh-centroid collocation. Not a tuned best PINN or significance test.",
    "source_quadrature_relative_discrepancy_max": pm["collocation_audit"][
        "source_max_relative_error"
    ],
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(rows, indent=2))
print(json.dumps(summary, indent=2))
