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
                "training_seconds": m.get("training_seconds", 709.944982),
                "run": name,
            }
        )
    a, b = manifests["raw_seed123"], manifests["ilu_seed123"]
    for k in (
        "data_sha256",
        "source_sha256",
        "loader_sha256",
        "initial_state_sha256",
        "seed",
        "config",
    ):
        assert a[k] == b[k], "Unmatched comparison: " + k
    archived, raw, ilu = [r["test_mse"] for r in rows]
    summary = {
        "raw_over_archived_ilu_test_mse": raw / archived,
        "raw_over_paired_ilu_test_mse": raw / ilu,
        "paired_ilu_has_lower_test_mse": ilu < raw,
        "paper_table2_mse": 1.70e-4,
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
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    labels = ["Archived ILU", "Raw residual", "Paired ILU"]
    for row, label in zip(rows, labels):
        values = loadmat(ROOT / "runs" / row["run"] / "E_test_pred.mat")[
            "sample_mse"
        ].ravel()
        axes[0].semilogy(np.arange(1, len(values) + 1), values, "o-", label=label)
    axes[0].set(
        xlabel="Test sample (original split order)", ylabel="Complex coefficient MSE"
    )
    axes[0].legend()
    axes[1].bar(labels, [r["test_mse"] for r in rows])
    axes[1].set(yscale="log", ylabel="Test MSE")
    fig.savefig(out / "test_mse_comparison.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, name, label in zip(
        axes, ["raw_seed123", "ilu_seed123"], ["Raw residual", "ILU residual"]
    ):
        values = np.loadtxt(
            ROOT / "runs" / name / "loss.csv", delimiter=",", skiprows=1
        )
        assert values.shape[0] == 1000 and np.isfinite(values).all()
        ax.semilogy(values[:, 0] + 1, values[:, 1], label="Train")
        ax.semilogy(values[:, 0] + 1, values[:, 2], label="Test")
        ax.set(xlabel="Epoch", ylabel=label + " loss", title=label)
        ax.legend()
    fig.savefig(out / "residual_convergence.png", dpi=200)
    plt.close(fig)
    text = [
        "Case 4 / SPP14: unpreconditioned residual ablation",
        "33 training / 8 testing; 1000 epochs; seed 123; same initial parameters.",
        "Metric: complex FEM coefficient MSE.",
    ]
    text += [
        f"{r['method']}: train MSE={r['train_mse']:.10g}, test MSE={r['test_mse']:.10g}, training seconds={r['training_seconds']:.3f}"
        for r in rows
    ]
    text += [
        f"Raw / paired ILU test MSE: {raw/ilu:.6g}",
        f"Raw / archived ILU test MSE: {raw/archived:.6g}",
        "Historical checkpoint was overwritten; archived predictions reproduce Table 2.",
        "Raw and ILU residual magnitudes are different objectives. Compare coefficient MSE.",
        "Random seed: 123.",
    ]
    (out / "report.txt").write_text("\n".join(text))
    print(json.dumps({"rows": rows, **summary}, indent=2))


if __name__ == "__main__":
    main()
