"""Plot fields."""

from pathlib import Path
import sys, json, csv, hashlib
import numpy as np
from scipy.io import loadmat
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "analysis_l2"))
from spatial_l2 import MatrixReader, element_values, integrate_linear

OUT = ROOT / "field_comparison"
fields = loadmat(
    OUT / "reconstructed_fields.mat", squeeze_me=True, struct_as_record=False
)["Fields"]
reader = MatrixReader(ROOT.parent / "SPP1/Train_data_SPP1_71.mat")
labels = ["FEM reference", "Archived ILU", "Paired ILU", "No preconditioning"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",
        "axes.linewidth": 0.6,
    }
)
metrics = []
for f in fields:
    coefficients = f.coefficients
    local, volume = element_values(reader, int(f.sample_id) - 1, "SPP1", coefficients)
    tri = f.tri.astype(int) - 1
    count = np.bincount(tri.ravel(), minlength=len(f.vertex))
    reconstructed = np.zeros((len(f.vertex), 2, 4), dtype=complex)
    for v in range(3):
        np.add.at(reconstructed, tri[:, v], local[:, v])
    reconstructed /= count[:, None, None]
    np.testing.assert_allclose(reconstructed[:, 0], f.Ex, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(reconstructed[:, 1], f.Ey, rtol=1e-11, atol=1e-12)
    for j in range(1, 4):
        integrals = integrate_linear(
            np.stack(
                [local[:, :, :, j] - local[:, :, :, 0], local[:, :, :, 0]], axis=-1
            ),
            volume,
        )
        metrics.append(
            dict(
                duty_cycle=float(f.duty_cycle),
                split=str(f.split),
                split_index=int(f.split_index),
                sample_id=int(f.sample_id),
                method=labels[j],
                coefficient_mse=float(
                    np.mean(abs(coefficients[:, j] - coefficients[:, 0]) ** 2)
                ),
                spatial_relative_l2=float(np.sqrt(integrals[0] / integrals[1])),
                maximum_magnitude_error=float(
                    np.max(abs(f.normE[:, j] - f.normE[:, 0]))
                ),
            )
        )
reader.close()


def draw(ax, f, values, vmin, vmax, cmap):
    # Model geometry is normalized by 976 nm (model parameters and duty cycle sweep).
    xy = f.vertex * 976
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1], f.tri.astype(int) - 1)
    artist = ax.tripcolor(
        tri, values, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True
    )
    ax.set_aspect("equal")
    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
    ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    return artist


for group, selected, title in [
    ("s2", fields, "Supplementary Fig. S2 — held-out test samples")
]:
    fig, axes = plt.subplots(2, 4, figsize=(12.8, 8.2), layout="constrained")
    for row, f in enumerate(selected):
        maximum = float(f.normE.max())
        for j, ax in enumerate(axes[row]):
            artist = draw(ax, f, f.normE[:, j], 0, maximum, "jet")
            ax.set_title(
                f"{labels[j]}\nw = {f.duty_cycle:.2f} ({f.split})", fontsize=10
            )
            if j > 0:
                metric = next(
                    x
                    for x in metrics
                    if x["duty_cycle"] == f.duty_cycle and x["method"] == labels[j]
                )
                ax.text(
                    0.5,
                    -0.24,
                    rf"Relative field $L^2$: {100*metric['spatial_relative_l2']:.2f}%",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=9,
                )
        cb = fig.colorbar(artist, ax=axes[row], shrink=0.79, pad=0.015)
        cb.set_label(r"$|E|$ — original field scale")
    fig.suptitle(title, fontsize=13)
    fig.savefig(OUT / f"{group}_fields.png", dpi=250)
    fig.savefig(OUT / f"{group}_fields.svg", dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 8.2), layout="constrained")
    for row, f in enumerate(selected):
        # Match outputE_v3.m: signed magnitude residual |E_ref| - |E_pred|.
        differences = f.normE[:, [0]] - f.normE[:, 1:]
        maximum = float(np.max(abs(differences)))
        for j, ax in enumerate(axes[row]):
            artist = draw(ax, f, differences[:, j], -maximum, maximum, "RdBu_r")
            ax.set_title(f"{labels[j+1]}\nw = {f.duty_cycle:.2f} ({f.split})")
        cb = fig.colorbar(artist, ax=axes[row], shrink=0.82, pad=0.015)
        cb.set_label(r"$|E_{ref}|-|E_{pred}|$")
    fig.suptitle(
        title + "\nMagnitude residuals; common scale within each duty cycle",
        fontsize=12,
    )
    fig.savefig(OUT / f"{group}_errors.png", dpi=250)
    fig.savefig(OUT / f"{group}_errors.svg", dpi=300)
    plt.close(fig)

with (OUT / "field_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
    w = csv.DictWriter(stream, fieldnames=list(metrics[0]))
    w.writeheader()
    w.writerows(metrics)
manifest = dict(
    duty_cycles=[0.60, 0.83],
    split="test",
    methods=labels,
    postprocessor="SPP1/post/GetExEy.m executed in MATLAB; full coefficients P @ reduced coefficients",
    periodic_mapping="Original Matrix{8,sample_id}, applied before field reconstruction and integration",
    reconstruction_crosscheck="Independent Python reconstruction and original MATLAB agree",
    coordinate_scale_nm=976,
    color_scale="Shared linear scale for all methods at each duty cycle",
    spatial_relative_l2="Exact element integration of complex vector-field error before node averaging",
    metrics=metrics,
)
(OUT / "plot_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(metrics, indent=2))
