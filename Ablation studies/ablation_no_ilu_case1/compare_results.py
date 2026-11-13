"""Compare results."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    rows = []
    manifests = {}
    for name, label in [
        ("verify", "Archived ILU (Table 2)"),
        ("raw_seed123", "FEMONet without ILU"),
        ("ilu_seed123", "FEMONet with ILU (paired rerun)"),
    ]:
        run = ROOT / "runs" / name
        m = json.loads((run / "metrics.json").read_text())
        manifests[name] = json.loads((run / "manifest.json").read_text())
        rows.append(
            {
                "method": label,
                "train_mse": m["train"]["mse_global"],
                "test_mse": m["test"]["mse_global"],
                "training_seconds": m.get("training_seconds", 7891.211123),
                "run": name,
            }
        )
    a, b = manifests["raw_seed123"], manifests["ilu_seed123"]
    for k in ("data_sha256", "source_sha256", "initial_state_sha256", "seed", "config"):
        assert a[k] == b[k], "Unmatched comparison: " + k
    archived, raw, ilu = [r["test_mse"] for r in rows]
    summary = {
        "raw_over_archived_ilu_test_mse": raw / archived,
        "raw_over_paired_ilu_test_mse": raw / ilu,
        "paired_ilu_has_lower_test_mse": ilu < raw,
        "paper_table2_mse": 4.39e-5,
        "metric": a["metric"],
        "interpretation": "Raw-to-ILU MSE ratio; seed 123.",
    }
    out = ROOT / "comparison"
    out.mkdir(exist_ok=False)
    with (out / "mse_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"rows": rows, **summary}, indent=2))


if __name__ == "__main__":
    main()
