"""Report results."""

import csv
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
data = loadmat(ROOT.parent / "SPP1/deepOnet_data_SPP1_71.mat")
rows = []
for name, label in [
    ("verify", "论文存档 ILU"),
    ("raw_seed123", "无预条件残差"),
    ("ilu_seed123", "同初始化 ILU 对照"),
]:
    metrics = json.loads((ROOT / "runs" / name / "metrics.json").read_text())
    for split in ("train", "test"):
        saved = loadmat(ROOT / "runs" / name / f"E_{split}_pred.mat")
        pred, truth = saved["E_pred"], data["Ez_" + split]
        lengths = data["coord_len_" + split].reshape(-1)
        assert pred.shape == truth.shape
        mask = np.arange(pred.shape[1])[None, :] < lengths[:, None]
        se = np.where(mask, np.abs(pred - truth) ** 2, 0).sum(axis=1)
        energy = np.where(mask, np.abs(truth) ** 2, 0).sum(axis=1)
        mse = float(se.sum() / lengths.sum())
        np.testing.assert_allclose(mse, metrics[split]["mse_global"], rtol=1e-10)
        np.testing.assert_allclose(
            se / lengths, saved["sample_mse"].ravel(), rtol=1e-10
        )
        rows.append(
            dict(
                method=label,
                run=name,
                split=split,
                mse=mse,
                relative_coefficient_l2=float(np.sqrt(se.sum() / energy.sum())),
                mean_sample_relative_coefficient_l2=float(
                    np.mean(np.sqrt(se / energy))
                ),
                training_seconds=metrics.get("training_seconds", 622.69709),
            )
        )
out = ROOT / "comparison"
with (out / "independent_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
test = {r["run"]: r for r in rows if r["split"] == "test"}
ratio = test["raw_seed123"]["mse"] / test["ilu_seed123"]["mse"]
lines = [
    "图 S2：周期金属光栅 无预条件残差消融结果",
    "",
    "相同的 57/14 数据划分、网络、seed=123 初始权重、1000 epochs 和优化器；仅移除 ILU 预条件。",
    "所有预测误差已在本机用原始 FEM 标签独立复算，并与集群结果一致。",
    "",
]
for r in rows:
    lines.append(
        f"{r['method']} / {r['split']}: MSE={r['mse']:.8e}, 系数相对 L2={r['relative_coefficient_l2']:.6%}"
    )
lines += [
    "",
    f"无预条件 / 配对 ILU 测试 MSE = {ratio:.6g} 倍。",
    f"无预条件 / 论文存档测试 MSE = {test['raw_seed123']['mse']/test['verify']['mse']:.6g} 倍。",
    "",
    "该比值大于 1 时支持 ILU 改善本次训练的精度，小于 1 时支持无预条件版本。",
    "这是单随机种子的受控消融结果，不代表统计显著性结论。",
    "MSE 与相对 L2 均针对复数 FEM 系数；不能将系数相对 L2 标为连续矢量场积分相对 L2。",
    "论文存档预测与图 S2 一致；集群预测文件属于不同历史运行，故配对重训是主要消融对照。",
    "不同残差目标的 loss 数量级不能直接用于判断预测精度。",
]
(out / "report_zh.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
