"""Plot fields."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "A-proceeding"
OUT = ROOT / "field_comparison"
OUT.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "svg.fonttype": "none",
    }
)

d = loadmat(DATA / "deepOnet_data_A1_all.mat")
mesh = loadmat(DATA / "Train_data_A1_all.mat", variable_names=["Matrix"])["Matrix"]
ids = loadmat(DATA / "idx_A1_all.mat")["testIdx"].ravel().astype(int) - 1
truth = d["Ez_test"]
lengths = d["coord_len_test"].ravel().astype(int)
pred = {
    name: loadmat(ROOT / "runs" / run / "E_test_pred.mat")["E_pred"]
    for name, run in [
        ("raw", "raw_seed123"),
        ("paired", "ilu_seed123"),
        ("archived", "verify"),
    ]
}
mask = np.arange(truth.shape[1])[None, :] < lengths[:, None]
mse = {
    key: np.where(mask, np.abs(v - truth) ** 2, 0).sum(axis=1) / lengths
    for key, v in pred.items()
}
order = np.argsort(mse["raw"], kind="stable")
quantiles = [0.1, 0.5, 0.9]
selected = [int(order[int(round(q * (len(order) - 1)))]) for q in quantiles]

# Validate mesh, ordering and independent metrics for ALL test samples.
for i, gid in enumerate(ids):
    n = lengths[i]
    np.testing.assert_array_equal(truth[i, :n], mesh[4, gid].ravel())
    np.testing.assert_array_equal(d["X_test"][i, :n], mesh[5, gid])
    tri = mesh[6, gid].astype(int) - 1
    assert tri.min() >= 0 and tri.max() < n
for key, run in [
    ("raw", "raw_seed123"),
    ("paired", "ilu_seed123"),
    ("archived", "verify"),
]:
    reported = json.loads((ROOT / "runs" / run / "metrics.json").read_text())["test"][
        "mse_global"
    ]
    np.testing.assert_allclose(
        np.sum(mse[key] * lengths) / lengths.sum(), reported, rtol=1e-12
    )

rows = []
for i in range(len(ids)):
    rows.append(
        {
            "test_sample_1based": i + 1,
            "original_sample_1based": int(ids[i] + 1),
            "nodes": int(lengths[i]),
            "mse_no_ilu": float(mse["raw"][i]),
            "mse_ilu_paired": float(mse["paired"][i]),
            "mse_ilu_table2": float(mse["archived"][i]),
            "selected_percentile": next(
                (int(q * 100) for q, j in zip(quantiles, selected) if i == j), ""
            ),
        }
    )
with (OUT / "sample_metrics.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def draw(reference, component):
    ref_label = "ILU (paired seed)" if reference == "paired" else "ILU (Table 2 model)"
    transform = np.real if component == "real" else np.abs
    symbol = r"$\mathrm{Re}\,E_z^s$" if component == "real" else r"$|E_z^s|$"
    all_fields = np.concatenate(
        [
            transform(v[i, : lengths[i]])
            for i in selected
            for v in [truth, pred["raw"], pred[reference]]
        ]
    )
    upper = float(np.max(np.abs(all_fields)))
    field_norm = Normalize(-upper if component == "real" else 0, upper)
    error_max = max(
        float(np.max(np.abs(pred[k][i, : lengths[i]] - truth[i, : lengths[i]])))
        for i in selected
        for k in ["raw", reference]
    )
    error_norm = Normalize(0, error_max)
    fig, axes = plt.subplots(3, 5, figsize=(17.5, 10.2), layout="constrained")
    fig.suptitle(
        "Case 1 | Scattered-field comparison: "
        + ("real component" if component == "real" else "magnitude"),
        fontsize=18,
        fontweight="bold",
    )
    titles = [
        "FEM reference",
        "FEMONet without ILU",
        ref_label,
        r"No ILU: $|E_{\rm pred}^s-E_{\rm FEM}^s|$",
        r"ILU: $|E_{\rm pred}^s-E_{\rm FEM}^s|$",
    ]
    for row, (q, i) in enumerate(zip(quantiles, selected)):
        n = lengths[i]
        gid = ids[i]
        xy = mesh[5, gid]
        triangles = mesh[6, gid].astype(int) - 1
        tri = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles)
        ys = [
            transform(truth[i, :n]),
            transform(pred["raw"][i, :n]),
            transform(pred[reference][i, :n]),
            np.abs(pred["raw"][i, :n] - truth[i, :n]),
            np.abs(pred[reference][i, :n] - truth[i, :n]),
        ]
        for col, (ax, z) in enumerate(zip(axes[row], ys)):
            art = ax.tripcolor(
                tri,
                z,
                shading="gouraud",
                cmap=(
                    ("RdBu_r" if component == "real" else "viridis")
                    if col < 3
                    else "magma"
                ),
                norm=field_norm if col < 3 else error_norm,
                rasterized=True,
            )
            if col < 3:
                field_artist = art
            else:
                error_artist = art
            ax.set_aspect("equal")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
            suffix = ""
            if col in [1, 3]:
                suffix = f"\nMSE = {mse['raw'][i]:.3e}"
            elif col in [2, 4]:
                suffix = f"\nMSE = {mse[reference][i]:.3e}"
            else:
                suffix = f"\nTest #{i+1} | source #{gid+1}"
            ax.set_title(titles[col] + suffix, pad=8)
            if row == 2:
                ax.set_xlabel(r"$x$ ($\mu$m)")
            if col == 0:
                ax.set_ylabel(f"P{int(q*100)} raw-MSE sample\n" + r"$y$ ($\mu$m)")
            else:
                ax.set_yticklabels([])
    cb = fig.colorbar(
        field_artist,
        ax=axes[:, :3],
        location="bottom",
        fraction=0.035,
        pad=0.045,
        aspect=65,
    )
    cb.set_label(symbol + " (V/m) — one shared scale for all field panels")
    cb = fig.colorbar(
        error_artist,
        ax=axes[:, 3:],
        location="bottom",
        fraction=0.035,
        pad=0.045,
        aspect=40,
    )
    cb.set_label("Complex field absolute error (V/m) — shared scale")
    fig.get_layout_engine().set(rect=(0, 0.055, 1, 0.95))
    fig.text(
        0.5,
        0.018,
        "Same test samples and original FEM triangles. Rows: nearest 10th / 50th / 90th percentile of no-ILU sample MSE.\n"
        "Errors include both real and imaginary components; MSE is the mean squared complex error over valid nodes.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#444444",
    )
    stem = OUT / f"field_{component}_{reference}"
    fig.savefig(str(stem) + ".png", dpi=180, facecolor="white")
    fig.savefig(str(stem) + ".svg", dpi=180, facecolor="white")
    plt.close(fig)
    return {
        "reference": reference,
        "component": component,
        "field_vmin": field_norm.vmin,
        "field_vmax": field_norm.vmax,
        "error_vmax": error_max,
    }


scales = [
    draw(ref, comp) for ref in ["paired", "archived"] for comp in ["real", "magnitude"]
]
(OUT / "plot_manifest.json").write_text(
    json.dumps(
        {
            "selection": "nearest rank to 10%, 50%, 90% of no-ILU test sample MSE",
            "selected_samples": [rows[i] for i in selected],
            "color_scales": scales,
            "field": "scattered electric field E_z^s, not total field",
            "rendering": "Gouraud shading on original FEM triangles; no spatial regridding",
            "validation": "All 312 sample coordinates and truth match mesh; all global MSE values reproduced",
        },
        indent=2,
    )
)
print(
    json.dumps(
        {"selected_samples": [rows[i] for i in selected], "output": str(OUT)}, indent=2
    )
)
