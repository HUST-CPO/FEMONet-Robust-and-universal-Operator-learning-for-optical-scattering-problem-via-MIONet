"""Compare results."""

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
names = ["adam1000", "adam3000", "adam5000_lbfgs500"]
ms = {
    name: json.loads((root / "runs" / name / "metrics.json").read_text())
    for name in names
}
man = {
    name: json.loads((root / "runs" / name / "manifest.json").read_text())
    for name in names
}
for key in [
    "initial_state_sha256",
    "baseline_runner_sha256",
    "source_sha256",
    "data_sha256",
    "collocation_sha256",
    "instance_sha256",
]:
    assert len({m[key] for m in man.values()}) == 1, key
ref = json.loads((root / "full_run_reference.json").read_text())
rows = [
    {
        "run": "28093 full training set, SAME instance",
        "checkpoint": "final",
        "adam_updates": 39000,
        "lbfgs_iterations": 0,
        "lbfgs_evaluations": 0,
        "mse": ref["same_instance_mse"],
        "relative_l2_nodes": ref["same_instance_relative_l2"],
        "optimization_seconds": None,
        "training_seconds_including_audits": ref["training_seconds"],
    }
]
for name, m in ms.items():
    history = list(csv.DictReader((root / "runs" / name / "loss.csv").open()))
    adam_seconds = float(
        next(r["seconds"] for r in history if r["stage"] == "adam_final")
    )
    for checkpoint in (["adam_final", "final"] if name == names[-1] else ["final"]):
        r = m[checkpoint]
        refine = checkpoint == "final"
        rows.append(
            {
                "run": name,
                "checkpoint": checkpoint,
                "adam_updates": m["adam_updates"],
                "lbfgs_iterations": m["lbfgs_iterations"] if refine else 0,
                "lbfgs_evaluations": m["lbfgs_function_evaluations"] if refine else 0,
                "mse": r["mse"],
                "relative_l2_nodes": r["relative_l2_nodes"],
                "optimization_seconds": m["optimization_seconds"] if refine else None,
                "training_seconds_including_audits": (
                    m["training_seconds_including_audits"] if refine else adam_seconds
                ),
            }
        )
with (out / "same_instance_comparison.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
summary = {
    "rows": rows,
    "checks": "Architecture/loss/data/collocation/initial weights hashes identical to job 28093",
    "sample_id": 1,
    "original_train_row": 449,
    "qualification": "One training geometry and one seed. Single-instance updates are 1000/3000/5000, versus 39000 for the full run. Effective batch size changes from up to 32 to 1. Longer-arm cosine horizons differ. L-BFGS is an explicit optimizer addition.",
    "timing": "Full run includes full-test validation. Single runs include independent spatial audits. Synchronized optimizer-only time is separately recorded for single runs.",
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
d = np.load(root / "instance.npz")
v = d["vertices"]
t = d["triangles"]
truth = d["truth"]
tri = mtri.Triangulation(v[:, 0], v[:, 1], t)
e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
lab = np.tile(d["domain"], 3)
e, ix = np.unique(e, axis=0, return_inverse=True)
lo = np.full(len(e), 100)
hi = np.zeros(len(e), int)
np.minimum.at(lo, ix, lab)
np.maximum.at(hi, ix, lab)
interface = v[e[lo != hi]]
fields = [truth]
titles = ["FEM reference"]
for name in names:
    f = loadmat(root / "runs" / name / "prediction_final.mat")
    np.testing.assert_array_equal(f["truth"].ravel(), truth)
    np.testing.assert_array_equal(f["vertices"], v)
    fields.append(f["prediction"].ravel())
    titles.append(name.replace("_", " + ") + f'\nMSE={ms[name]["final"]["mse"]:.3e}')
plt.rcParams.update(
    {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}
)
fig, axs = plt.subplots(2, 4, figsize=(14, 7.5), layout="constrained")
for r, part in enumerate((np.real, np.imag)):
    bound = max(abs(part(z)).max() for z in fields)
    for c, (z, title) in enumerate(zip(fields, titles)):
        ax = axs[r, c]
        im = ax.tripcolor(
            tri,
            part(z),
            norm=Normalize(-bound, bound),
            cmap="RdBu_r",
            shading="gouraud",
            rasterized=True,
        )
        ax.add_collection(LineCollection(interface, colors="black", linewidths=0.7))
        ax.set(
            xlim=(-1, 1),
            ylim=(-1, 1),
            aspect="equal",
            xticks=[-1, 0, 1],
            yticks=[-1, 0, 1],
            xlabel="x (µm)",
            ylabel=["Re(u)", "Im(u)"][r] + "\ny (µm)",
            title=title,
        )
    fig.colorbar(im, ax=axs[r, :], pad=0.01, shrink=0.8)
fig.suptitle(
    "Job-28093 DeepONet and loss · single training sample ID 1\nAll parameters trainable; same initialization; predeclared endpoints",
    fontsize=15,
)
for ext in ("png", "svg"):
    fig.savefig(out / f"single_deeponet_fields.{ext}", dpi=200)
plt.close(fig)
fig, axs = plt.subplots(2, 3, figsize=(11, 7.8), layout="constrained")
pred = fields[-1]
for r, part in enumerate((np.real, np.imag)):
    bound = max(abs(part(truth)).max(), abs(part(pred)).max())
    for c, z in enumerate((part(truth), part(pred), abs(part(pred - truth)))):
        ax = axs[r, c]
        im = ax.tripcolor(
            tri,
            z,
            norm=Normalize(-bound, bound) if c < 2 else Normalize(0, z.max()),
            cmap="RdBu_r" if c < 2 else "magma",
            shading="gouraud",
            rasterized=True,
        )
        ax.add_collection(LineCollection(interface, colors="black", linewidths=0.7))
        ax.set(
            xlim=(-1, 1),
            ylim=(-1, 1),
            aspect="equal",
            xticks=[-1, 0, 1],
            yticks=[-1, 0, 1],
            xlabel="x (µm)",
            ylabel="y (µm)",
            title=[
                "FEM reference",
                "DeepONet PINN: Adam + L-BFGS",
                "Absolute component error",
            ][c]
            + "\n"
            + ["Real part", "Imaginary part"][r],
        )
        if c == 1:
            fig.colorbar(im, ax=axs[r, :2], pad=0.01, shrink=0.8)
        if c == 2:
            fig.colorbar(im, ax=ax, pad=0.01, shrink=0.8)
fig.suptitle(
    "Same job-28093 architecture and loss · single instance ID 1\n"
    + f'Adam 5000 updates + L-BFGS; MSE={ms[names[-1]]["final"]["mse"]:.3e}',
    fontsize=14,
)
for ext in ("png", "svg"):
    fig.savefig(out / f"refined_field_detail.{ext}", dpi=200)
plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
for name in names:
    data = list(csv.DictReader((root / "runs" / name / "loss.csv").open()))
    a = [r for r in data if r["stage"] == "Adam"]
    ax.semilogy(
        [int(r["step"]) for r in a],
        [float(r["mse"]) for r in a],
        label=name + " (Adam stage)",
    )
ax.axhline(
    ref["same_instance_mse"],
    color="black",
    linestyle="--",
    label="28093 final: same instance, 39000 updates",
)
ax.scatter(
    [5000],
    [ms[names[-1]]["final"]["mse"]],
    marker="*",
    s=120,
    color="red",
    label="After L-BFGS (additional work)",
)
ax.set(
    xlabel="Adam optimizer updates (L-BFGS work is additional)",
    ylabel="Same-instance complex field MSE",
    title="Endpoint budgets and Adam histories",
)
ax.grid(alpha=0.25)
ax.legend(fontsize=8)
fig.savefig(out / "mse_by_adam_updates.png", dpi=200)
plt.close(fig)
print(json.dumps(summary, indent=2))
