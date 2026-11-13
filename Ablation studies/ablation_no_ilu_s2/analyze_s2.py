"""Analyze s2."""

from pathlib import Path
import sys, json, csv
import numpy as np
from scipy.io import loadmat
from scipy.stats import norm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "analysis_l2"))
from spatial_l2 import MatrixReader, element_values, integrate_linear

OUT = ROOT / "comparison"
data = loadmat(ROOT.parent / "SPP1/deepOnet_data_SPP1_71.mat")
ids = loadmat(ROOT.parent / "SPP1/idx_SPP1_71.mat")
groups = [
    ("verify", "Archived ILU"),
    ("ilu_seed123", "Paired ILU"),
    ("raw_seed123", "No preconditioning"),
]
rows = []
fits = []
spatial = []
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
for ax, split in zip(axes, ["train", "test"]):
    distributions = []
    for run, label in groups:
        saved = loadmat(ROOT / "runs" / run / f"E_{split}_pred.mat")
        n = data["coord_len_" + split].ravel().astype(np.int64)
        mask = np.arange(saved["E_pred"].shape[1])[None, :] < n[:, None]
        mse = (
            np.where(mask, abs(saved["E_pred"] - data["Ez_" + split]) ** 2, 0).sum(
                axis=1
            )
            / n
        )
        np.testing.assert_allclose(mse, saved["sample_mse"].ravel(), rtol=1e-10)
        assert np.all(mse > 0) and np.isfinite(mse).all()
        log = np.log10(mse)
        distributions.append(log)
        mu = float(log.mean())
        sigma = float(log.std(ddof=1))
        fits.append(
            dict(
                method=label,
                split=split,
                n=len(mse),
                mu_log10_mse=mu,
                sigma_log10_mse=sigma,
            )
        )
        for i, value in enumerate(mse):
            sid = int(ids[split + "Idx"].ravel()[i])
            rows.append(
                dict(
                    method=label,
                    split=split,
                    split_index=i + 1,
                    sample_id=sid,
                    duty_cycle=0.2 + 0.01 * (sid - 1),
                    mse=float(value),
                )
            )
    bins = np.linspace(
        min(x.min() for x in distributions) - 0.15,
        max(x.max() for x in distributions) + 0.15,
        12,
    )
    x = np.linspace(bins[0], bins[-1], 400)
    for (_, label), values, color in zip(
        groups, distributions, ["#356eb8", "#259252", "#df7620"]
    ):
        mu = values.mean()
        sigma = values.std(ddof=1)
        ax.hist(
            values, bins=bins, density=True, histtype="step", linewidth=1.5, color=color
        )
        ax.plot(
            x,
            norm.pdf(x, mu, sigma),
            color=color,
            label=f"{label}: $\mu$={mu:.2f}, $\sigma$={sigma:.2f}",
        )
    ax.set(
        xlabel=r"$\log_{10}$(per-sample MSE)",
        ylabel="Density",
        title=f"{split.capitalize()} samples (n={len(distributions[0])})",
    )
    ax.legend(fontsize=8)
fig.savefig(OUT / "s2_mse_distributions.png", dpi=220)
fig.savefig(OUT / "s2_mse_distributions.svg")
plt.close(fig)

reader = MatrixReader(ROOT.parent / "SPP1/Train_data_SPP1_71.mat")
summary = []
for run, label in groups:
    pred = loadmat(ROOT / "runs" / run / "E_test_pred.mat")["E_pred"]
    group = []
    for i, sid in enumerate(ids["testIdx"].ravel().astype(int)):
        sid = int(sid)
        n = int(data["coord_len_test"].ravel()[i])
        truth = data["Ez_test"][i, :n]
        np.testing.assert_allclose(
            reader.get(4, sid - 1).ravel(), truth, rtol=1e-11, atol=1e-12
        )
        local, vol = element_values(
            reader, sid - 1, "SPP1", np.stack([pred[i, :n] - truth, truth], axis=1)
        )
        error, reference = integrate_linear(local, vol)
        row = dict(
            method=label,
            split_index=i + 1,
            sample_id=sid,
            duty_cycle=0.2 + 0.01 * (sid - 1),
            error_integral=float(error),
            reference_integral=float(reference),
            spatial_relative_l2=float(np.sqrt(error / reference)),
        )
        spatial.append(row)
        group.append(row)
    summary.append(
        dict(
            method=label,
            n_samples=len(group),
            spatial_relative_l2_pooled=float(
                np.sqrt(
                    sum(r["error_integral"] for r in group)
                    / sum(r["reference_integral"] for r in group)
                )
            ),
            mean_sample_spatial_relative_l2=float(
                np.mean([r["spatial_relative_l2"] for r in group])
            ),
        )
    )
reader.close()
for name, content in [
    ("sample_mse", rows),
    ("gaussian_fit", fits),
    ("spatial_per_sample", spatial),
    ("spatial_summary", summary),
]:
    with (OUT / (name + ".csv")).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(content[0]))
        w.writeheader()
        w.writerows(content)
    (OUT / (name + ".json")).write_text(json.dumps(content, indent=2))
with (OUT / "report_zh.txt").open("a", encoding="utf-8") as f:
    f.write(
        "\n\n全部14个测试样本的周期矢量电场积分误差（先恢复P映射；不同于系数L2）：\n"
    )
    for row in summary:
        f.write(
            f"{row['method']}: 合并相对 L2={row['spatial_relative_l2_pooled']:.4%}\n"
        )
print(json.dumps(summary, indent=2))
