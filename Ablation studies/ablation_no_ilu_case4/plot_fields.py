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
reader = MatrixReader(ROOT.parent / "SPP14/Train_data_SPP14_41.mat")
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
    local, volume = element_values(reader, int(f.sample_id) - 1, "4", coefficients)
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
                wavelength_nm=int(f.wavelength_nm),
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
    # Model geometry is normalized by 500 nm (model parameters and wavelength sweep).
    xy = f.vertex * 500
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
    (
        "paper_wavelengths",
        fields[:2],
        "Paper Fig. 6 wavelengths — training samples in the saved split",
    ),
    (
        "test_wavelengths",
        fields[2:],
        "Held-out test samples nearest to the paper wavelengths",
    ),
]:
    fig, axes = plt.subplots(2, 4, figsize=(12.8, 8.2), layout="constrained")
    for row, f in enumerate(selected):
        maximum = float(f.normE.max())
        for j, ax in enumerate(axes[row]):
            artist = draw(ax, f, f.normE[:, j], 0, maximum, "jet")
            ax.set_title(
                f"{labels[j]}\n{int(f.wavelength_nm)} nm ({f.split})", fontsize=10
            )
            if j > 0:
                metric = next(
                    x
                    for x in metrics
                    if x["wavelength_nm"] == f.wavelength_nm
                    and x["method"] == labels[j]
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
            ax.set_title(f"{labels[j+1]}\n{int(f.wavelength_nm)} nm ({f.split})")
        cb = fig.colorbar(artist, ax=axes[row], shrink=0.82, pad=0.015)
        cb.set_label(r"$|E_{ref}|-|E_{pred}|$")
    fig.suptitle(
        title + "\nMagnitude residuals; common scale within each wavelength",
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
    paper_wavelengths_nm=[535, 655],
    actual_split="train",
    nearest_test_wavelengths_nm=[530, 650],
    all_test_wavelengths_nm=[610, 650, 560, 620, 625, 530, 505, 600],
    methods=labels,
    postprocessor="SPP14/post/GetExEy.m, directly executed in MATLAB R2023b",
    reconstruction_crosscheck="Passed: independent Python element basis evaluation and vertex averaging match MATLAB",
    magnitude="sqrt(abs(Ex)^2 + abs(Ey)^2), unchanged original field scale",
    residual="abs(E_ref vector) - abs(E_pred vector), as outputE_v3.m; signed magnitude difference",
    color_scale="Common linear scale across methods for each wavelength; no independent normalization or clipping",
    coordinate_scale_nm=500,
    spatial_relative_l2="Exact triangle integration of complex vector-field error, before vertex averaging; full mesh",
    checkpoint_selection="Final epoch 1000; no retraining or test-based checkpoint selection",
    inputs_sha256={
        str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            ROOT.parent / "SPP14/post/GetExEy.m",
            OUT / "reconstructed_fields.mat",
        ]
    },
    metrics=metrics,
)
(OUT / "plot_manifest.json").write_text(json.dumps(manifest, indent=2))
(OUT / "说明.txt").write_text(
    "论文图6标注535/655 nm，但保存的划分中二者属于训练集（训练样本6、7）。\n"
    "实际测试集相邻波长为530/650 nm（测试样本6、2）。两套图分别保存并明确标注。\n"
    "场图依次为FEM真值、论文存档ILU、本次配对ILU、无预条件残差。相同波长下共用色标。\n"
    "场重建直接调用项目原GetExEy.m，并与独立Python边元重建交叉核验一致。\n"
    "误差图采用原绘图脚本的有符号幅值差 |E_ref|-|E_pred|；并非复矢量误差的模。\n"
    "图中相对L2是全域复矢量电场的精确三角形积分误差，未进行节点平均；与上一轮系数相对L2不同。\n"
    "PNG用于预览，SVG保留可编辑文字与坐标，场着色以300dpi嵌入。\n",
    encoding="utf-8",
)
print(json.dumps(metrics, indent=2))
