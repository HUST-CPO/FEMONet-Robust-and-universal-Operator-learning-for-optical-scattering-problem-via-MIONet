"""Report results."""

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
modes = ["manufactured", "fixed", "resample1", "resample10", "ilu"]
metrics = {
    m: json.loads((root / "runs" / f"{m}_seed123" / "metrics.json").read_text())
    for m in modes
}
manifests = {
    m: json.loads((root / "runs" / f"{m}_seed123" / "manifest.json").read_text())
    for m in modes
}
for key in ("initial_hash", "instance_sha256", "network", "steps", "lbfgs"):
    assert len({str(v[key]) for v in manifests.values()}) == 1, key
rows = []
for m in modes:
    for checkpoint in ("final", "best_physics"):
        rows.append(
            {
                "mode": m,
                "checkpoint": checkpoint,
                **metrics[m][checkpoint],
                "training_seconds": metrics[m]["training_seconds"],
                "best_independent_physics": metrics[m]["best_physics_score"],
            }
        )
with (out / "all_results.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
selected = min(
    ["fixed", "resample1", "resample10"], key=lambda m: metrics[m]["best_physics_score"]
)
summary = {
    "selected_pinn_by_independent_physics": selected,
    "selected_pinn": metrics[selected]["best_physics"],
    "matched_ilu_final": metrics["ilu"]["final"],
    "manufactured": metrics["manufactured"]["best_physics"],
    "manufactured_mse_below_1e_minus3": metrics["manufactured"]["best_physics"]["mse"]
    < 1e-3,
    "pinn_to_ilu_mse_ratio": metrics[selected]["best_physics"]["mse"]
    / metrics["ilu"]["final"]["mse"],
    "source_audit": json.loads((root / "source_audit.json").read_text()),
    "limitation": "One development geometry and one seed. Different coordinate-only architecture from operator experiment. No claim about full operator convergence.",
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
d = np.load(root / "instance.npz")
v = d["vertices"]
t = d["triangles"]
u = d["truth"]
tri = mtri.Triangulation(v[:, 0], v[:, 1], t)
edges = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
labels = np.tile(d["domain"], 3)
e, ix = np.unique(edges, axis=0, return_inverse=True)
lo = np.full(len(e), 100)
hi = np.zeros(len(e), int)
np.minimum.at(lo, ix, labels)
np.maximum.at(hi, ix, labels)
interface = v[e[lo != hi]]
p = loadmat(root / "runs" / f"{selected}_seed123" / "prediction_best_physics.mat")[
    "prediction"
].ravel()
fem = loadmat(root / "runs/ilu_seed123/prediction_final.mat")["prediction"].ravel()
plt.rcParams.update(
    {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}
)
fig, axs = plt.subplots(2, 4, figsize=(14, 7.8), layout="constrained")
for r, component in enumerate((np.real, np.imag)):
    fields = [component(z) for z in (u, fem, p)]
    bound = max(abs(z).max() for z in fields)
    for c, z in enumerate(fields + [abs(component(p - u))]):
        ax = axs[r, c]
        im = ax.tripcolor(
            tri,
            z,
            shading="gouraud",
            cmap="RdBu_r" if c < 3 else "magma",
            norm=Normalize(-bound, bound) if c < 3 else Normalize(0, z.max()),
            rasterized=True,
        )
        ax.add_collection(LineCollection(interface, colors="black", linewidths=0.7))
        ax.set(
            aspect="equal",
            xlabel="x (µm)",
            ylabel="y (µm)",
            xticks=[-1, 0, 1],
            yticks=[-1, 0, 1],
            title=[
                "FEM reference",
                "Matched ILU-FEM network",
                f"PINN ({selected})",
                "PINN absolute error",
            ][c]
            + "\n"
            + ["Real part", "Imaginary part"][r],
        )
        if c == 2:
            fig.colorbar(im, ax=axs[r, :3], shrink=0.8, pad=0.01)
        if c == 3:
            fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
fig.suptitle(
    "Single training instance · original ID 1\n"
    + f'PINN MSE={summary["selected_pinn"]["mse"]:.3e}; matched ILU-FEM MSE={summary["matched_ilu_final"]["mse"]:.3e}',
    fontsize=15,
)
for ext in ("png", "svg"):
    fig.savefig(out / f"selected_field_comparison.{ext}", dpi=220)
plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
for m in modes:
    if m == "manufactured":
        continue
    a = list(csv.DictReader((root / "runs" / f"{m}_seed123" / "loss.csv").open()))
    ax.semilogy(
        [float(x["seconds"]) / 60 for x in a],
        [float(x["validation_sum"]) for x in a],
        label=m,
    )
ax.set(
    xlabel="Training wall time (min)",
    ylabel="Independent normalized PDE + boundary residual",
    title="Common strong-form validation metric (not field MSE)",
)
ax.grid(alpha=0.2)
ax.legend()
fig.savefig(out / "independent_physics_history.png", dpi=200)
plt.close(fig)
print(json.dumps(summary, indent=2))
