"""Plot fields."""

import csv
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
DATA = ROOT.parent / "A-proceeding"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def main():
    ref = loadmat(
        DATA / "deepOnet_data_A1_all.mat",
        variable_names=["X_test", "Ez_test", "coord_len_test"],
    )
    mesh = loadmat(
        DATA / "Train_data_A1_all.mat",
        variable_names=["Matrix", "Nodes_test", "Elements_test", "Domains_test"],
    )
    ids = loadmat(DATA / "idx_A1_all.mat")["testIdx"].ravel().astype(int)
    stored = loadmat(ROOT / "runs/pinn_seed123/E_test_pred.mat")
    pred = stored["E_pred"]
    truth = ref["Ez_test"]
    lengths = ref["coord_len_test"].ravel().astype(int)
    np.testing.assert_array_equal(stored["coord_len"].ravel(), lengths)
    mask = np.arange(pred.shape[1])[None, :] < lengths[:, None]
    se = np.where(mask, abs(pred - truth) ** 2, 0).sum(axis=1)
    mse = se / lengths
    np.testing.assert_allclose(mse, stored["sample_mse"].ravel(), rtol=1e-12)
    metrics = json.loads((ROOT / "runs/pinn_seed123/metrics.json").read_text())
    np.testing.assert_allclose(
        se.sum() / lengths.sum(), metrics["test"]["mse_global"], rtol=1e-12
    )
    rows = [
        {
            "test_row_1based": i + 1,
            "sample_id": int(ids[i]),
            "valid_dofs": int(lengths[i]),
            "complex_field_mse": float(mse[i]),
        }
        for i in range(len(ids))
    ]
    with (OUT / "test_sample_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def get(i):
        n = lengths[i]
        v = ref["X_test"][i, :n]
        t = mesh["Matrix"][6, ids[i] - 1].astype(int) - 1
        np.testing.assert_array_equal(v, mesh["Matrix"][5, ids[i] - 1])
        np.testing.assert_array_equal(
            truth[i, :n], mesh["Matrix"][4, ids[i] - 1].ravel()
        )
        np.testing.assert_array_equal(v, mesh["Nodes_test"][0, i])
        np.testing.assert_array_equal(t + 1, mesh["Elements_test"][0, i])
        domain = mesh["Domains_test"][0, i].ravel()
        edges = np.sort(
            np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1
        )
        labels = np.tile(domain, 3)
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        low = np.full(len(unique), 100)
        high = np.zeros(len(unique), dtype=int)
        np.minimum.at(low, inverse, labels)
        np.maximum.at(high, inverse, labels)
        interface = v[unique[low != high]]
        return (
            mtri.Triangulation(v[:, 0], v[:, 1], t),
            interface,
            truth[i, :n],
            pred[i, :n],
        )

    def panel(ax, tri, interface, z, norm, cmap, title):
        artist = ax.tripcolor(
            tri, z, shading="gouraud", cmap=cmap, norm=norm, rasterized=True
        )
        ax.add_collection(
            LineCollection(interface, colors="#222222", linewidths=0.65, alpha=0.8)
        )
        ax.set(
            xlim=(-1, 1),
            ylim=(-1, 1),
            aspect="equal",
            xlabel="x (µm)",
            ylabel="y (µm)",
            title=title,
        )
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        return artist

    i = 273
    tri, interface, u, p = get(i)
    fig, axs = plt.subplots(3, 3, figsize=(12, 11), layout="constrained")
    columns = ["FEM reference", "PINN prediction", "Absolute error"]
    specs = [
        (u.real, p.real, abs(p.real - u.real), "Real part", "RdBu_r"),
        (u.imag, p.imag, abs(p.imag - u.imag), "Imaginary part", "RdBu_r"),
        (abs(u), abs(p), abs(p - u), "Magnitude", "viridis"),
    ]
    for r, (a, b, e, label, cmap) in enumerate(specs):
        bound = max(abs(a).max(), abs(b).max())
        norm = Normalize(-bound if r < 2 else 0, bound)
        for c, z in enumerate((a, b)):
            im = panel(
                axs[r, c], tri, interface, z, norm, cmap, columns[c] + " · " + label
            )
        fig.colorbar(
            im,
            ax=axs[r, :2],
            shrink=0.8,
            pad=0.02,
            label="Scattered field / incident amplitude",
        )
        title = (
            "|Re(PINN − FEM)|"
            if r == 0
            else "|Im(PINN − FEM)|" if r == 1 else "|PINN − FEM| (complex field)"
        )
        im = panel(axs[r, 2], tri, interface, e, Normalize(0, e.max()), "magma", title)
        fig.colorbar(im, ax=axs[r, 2], shrink=0.8, pad=0.02, label="Absolute error")
    fig.suptitle(
        f"Case 1 · PINN scattered-field comparison\nTest sample 274 (original ID {ids[i]}; Fig. S1A sample) · MSE = {mse[i]:.6f}",
        fontsize=16,
    )
    fig.get_layout_engine().set(rect=(0, 0.035, 1, 0.965))
    fig.text(
        0.5,
        0.012,
        "Shared FEM/PINN color scale within each row. Black curves: material interfaces. Original FEM mesh; no smoothing filter.",
        ha="center",
        fontsize=10,
    )
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"pinn_field_test274.{ext}", dpi=220)
    plt.close(fig)

    order = np.argsort(mse, kind="stable")
    selected = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    items = [get(j) for j in selected]
    field_max = max(max(abs(u).max(), abs(p).max()) for _, _, u, p in items)
    error_max = max(abs(p - u).max() for _, _, u, p in items)
    fig, axs = plt.subplots(3, 3, figsize=(12, 11), layout="constrained")
    for r, (j, item) in enumerate(zip(selected, items)):
        tri, interface, u, p = item
        for c, z in enumerate((abs(u), abs(p), abs(p - u))):
            im = panel(
                axs[r, c],
                tri,
                interface,
                z,
                Normalize(0, field_max if c < 2 else error_max),
                "viridis" if c < 2 else "magma",
                ["FEM |u|", "PINN |u|", "Complex error |PINN − FEM|"][c],
            )
            if c == 1:
                fig.colorbar(
                    im, ax=axs[r, :2], shrink=0.8, pad=0.02, label="Field amplitude"
                )
            if c == 2:
                fig.colorbar(
                    im, ax=axs[r, 2], shrink=0.8, pad=0.02, label="Absolute error"
                )
        axs[r, 0].set_ylabel(
            ["Lowest MSE", "Median rank (157/312)", "Highest MSE"][r]
            + f"\nTest {j+1}, ID {ids[j]}\nMSE={mse[j]:.5f}\ny (µm)"
        )
    fig.suptitle(
        "PINN test-set overview · samples ranked by complex field MSE", fontsize=16
    )
    fig.get_layout_engine().set(rect=(0, 0.035, 1, 0.965))
    fig.text(
        0.5,
        0.012,
        "One shared amplitude scale and one shared error scale across all selected samples. Black curves: material interfaces.",
        ha="center",
        fontsize=10,
    )
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"pinn_field_error_overview.{ext}", dpi=220)
    plt.close(fig)
    provenance = {
        "detail": rows[i],
        "overview": [rows[j] for j in selected],
        "test_global_mse": float(se.sum() / lengths.sum()),
        "field": "complex scattered Ez, normalized by unit incident amplitude",
        "display": "linear vertex-color interpolation on saved FEM triangulation; no regular-grid resampling",
        "error": "nodewise component absolute error in real/imaginary panels; complex magnitude error in amplitude panels",
        "selection": "test row 274 matches existing S1A record; overview min/upper-median/max valid-DOF per-sample MSE",
    }
    (OUT / "field_plot_metadata.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
