"""Compare full results."""

import csv, json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
COUNTS = [10, 30, 100, 300, 1000, 1246]


def main():
    rows = []
    manifests = {}
    metrics = {}
    for mode, n in [("pi", 0)] + [(m, n) for m in ("dd", "hybrid") for n in COUNTS]:
        name = f"{mode}_n{n}_seed123"
        d = ROOT / "runs" / name
        a = json.loads((d / "manifest.json").read_text())
        b = json.loads((d / "metrics.json").read_text())
        assert b["updates"] == 1000 * (((1246 if mode == "pi" else n) + 31) // 32)
        manifests[mode, n] = a
        metrics[mode, n] = b
        rows.append(
            dict(
                method=mode,
                n_label=n,
                n_physics=a["n_physics"],
                epochs=1000,
                updates=b["updates"],
                seed=a["seed"],
                test_mse=b["test"]["mse_global"],
                train_mse=b["train"]["mse_global"],
                training_seconds=b["training_seconds"],
                setup_seconds=b["setup_seconds"],
            )
        )
    base = manifests["pi", 0]
    for a in manifests.values():
        for k in (
            "config",
            "seed",
            "initial_state_sha256",
            "data_sha256",
            "source_sha256",
            "implementation_sha256",
        ):
            assert a[k] == base[k], k
    for n in COUNTS:
        assert manifests["dd", n]["labels"] == manifests["hybrid", n]["labels"]
        assert (
            metrics["dd", n]["label_order_sha256"]
            == metrics["hybrid", n]["label_order_sha256"]
        )
        assert (
            metrics["dd", n]["label_order_sha256"]
            == metrics["hybrid", n]["physical_order_sha256"]
        )
    out = ROOT / "comparison_full"
    out.mkdir(exist_ok=False)
    with (out / "mse_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    pi = metrics["pi", 0]["test"]["mse_global"]
    paper = 4.390900391795585e-5
    summary = dict(
        pi_test_mse=pi,
        table2_mse=paper,
        single_seed=True,
        protocol="1000 epochs; Hybrid uses N physical conditions; PI uses 1246; unequal update budgets",
        minimum_tested_labels={},
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for mode, color, label in [
        ("dd", "#2388cf", "Data-driven"),
        ("hybrid", "#bc852a", "Hybrid"),
    ]:
        ys = [metrics[mode, n]["test"]["mse_global"] for n in COUNTS]
        ax.plot(COUNTS, ys, "o-", label=label, color=color)
        hits = [n for n, y in zip(COUNTS, ys) if y <= pi]
        summary["minimum_tested_labels"][mode] = (
            min(hits) if hits else "Not reached through 1246 labels"
        )
    ax.axhline(
        pi,
        color="#109e78",
        linestyle="--",
        label="FEMONet-ILU (0 labels, 1246 conditions)",
    )
    ax.axhline(
        paper, color="#777777", linestyle=":", label="Table 2 archived reference"
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Number of paired FEM solution labels",
        ylabel="Test complex-field MSE",
    )
    ax.set_xticks(COUNTS, labels=[str(n) for n in COUNTS])
    ax.tick_params(axis="x", labelrotation=25)
    ax.minorticks_off()
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "label_efficiency.png", dpi=250)
    fig.savefig(out / "label_efficiency.pdf")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
