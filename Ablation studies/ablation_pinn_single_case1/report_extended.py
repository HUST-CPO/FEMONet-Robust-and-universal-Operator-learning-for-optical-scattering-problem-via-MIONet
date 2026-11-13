"""Report extended."""

import csv, json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

root = Path(__file__).resolve().parent
out = root / "comparison"
out.mkdir(exist_ok=True)
rows = []
for mode, tag in [
    ("manufactured", ""),
    ("fixed", ""),
    ("resample1", ""),
    ("resample10", ""),
    ("ilu", ""),
    ("fixed", "_extended"),
    ("resample1", "_extended"),
    ("ilu", "_extended"),
]:
    run = root / "runs" / f"{mode}_seed123{tag}"
    a = json.loads((run / "manifest.json").read_text())
    b = json.loads((run / "metrics.json").read_text())
    for ck in ("final", "best_physics"):
        rows.append(
            {
                "run": run.name,
                "mode": mode,
                "adam_steps": a["steps"],
                "lbfgs_max_iterations": a["lbfgs"],
                "checkpoint": ck,
                **b[ck],
                "training_seconds": b["training_seconds"],
                "best_physics_score": b["best_physics_score"],
            }
        )
with (out / "all_budgets_results.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
finals = {r["run"]: r for r in rows if r["checkpoint"] == "final"}
summary = {
    "final_checkpoints": finals,
    "fixed_over_ilu_mse_initial": finals["fixed_seed123"]["mse"]
    / finals["ilu_seed123"]["mse"],
    "fixed_over_ilu_mse_extended": finals["fixed_seed123_extended"]["mse"]
    / finals["ilu_seed123_extended"]["mse"],
    "scope": "One training geometry, one seed, coordinate-only network. Final checkpoints reported for all settings; no choice by desired ranking.",
    "selection_caveat": "Independent residual selection and field-error ranking disagree in the first round. Both final and physics-selected results are retained.",
    "timing_scope": "Optimization plus validation and export; excludes FEM assembly and ILU setup.",
}
(out / "extended_summary.json").write_text(json.dumps(summary, indent=2))
d = np.load(root / "instance.npz")
v = d["vertices"]
t = d["triangles"]
tri = mtri.Triangulation(v[:, 0], v[:, 1], t)
ed = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
labels = np.tile(d["domain"], 3)
ed, ix = np.unique(ed, axis=0, return_inverse=True)
lo = np.full(len(ed), 100)
hi = np.zeros(len(ed), int)
np.minimum.at(lo, ix, labels)
np.maximum.at(hi, ix, labels)
interface = v[ed[lo != hi]]
plt.rcParams.update(
    {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}
)
for tag in ("", "_extended"):
    names = ["ilu", "fixed", "resample1"] + (["resample10"] if not tag else [])
    fields = [d["truth"]] + [
        loadmat(root / "runs" / f"{m}_seed123{tag}" / "prediction_final.mat")[
            "prediction"
        ].ravel()
        for m in names
    ]
    titles = ["FEM reference"] + [
        f'{m}\nMSE={finals[f"{m}_seed123{tag}"]["mse"]:.3e}' for m in names
    ]
    fig, axs = plt.subplots(
        2, len(fields), figsize=(3.2 * len(fields), 7.4), layout="constrained"
    )
    for r, part in enumerate((np.real, np.imag)):
        bound = max(abs(part(z)).max() for z in fields)
        for c, z in enumerate(fields):
            ax = axs[r, c]
            im = ax.tripcolor(
                tri,
                part(z),
                cmap="RdBu_r",
                norm=Normalize(-bound, bound),
                shading="gouraud",
                rasterized=True,
            )
            ax.add_collection(LineCollection(interface, colors="black", linewidths=0.7))
            ax.set(
                xlim=(-1, 1),
                ylim=(-1, 1),
                aspect="equal",
                xlabel="x (µm)",
                ylabel=["Re(u)", "Im(u)"][r] + "\ny (µm)",
                xticks=[-1, 0, 1],
                yticks=[-1, 0, 1],
                title=titles[c],
            )
        fig.colorbar(im, ax=axs[r, :], shrink=0.8, pad=0.01)
    fig.suptitle(
        "Single-instance final checkpoints · training sample ID 1\n5000 Adam updates + "
        + ("3000" if tag else "500")
        + " maximum L-BFGS iterations",
        fontsize=15,
    )
    for ext in ("png", "svg"):
        fig.savefig(out / f"all_final_fields{tag}.{ext}", dpi=200)
    plt.close(fig)
print(json.dumps(summary, indent=2))
