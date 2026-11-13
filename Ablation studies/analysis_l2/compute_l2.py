"""Reproduce complex coefficient errors; no training or source mutation."""

from pathlib import Path
import csv, json, hashlib
import numpy as np
import h5py
from scipy.io import loadmat
from scipy.sparse import csc_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def read_h5(h, obj):
    if isinstance(obj, h5py.Group):
        if "MATLAB_sparse" in obj.attrs:
            a = read_h5(h, obj["data"]).ravel()
            return csc_matrix(
                (a, obj["ir"][()].astype(int), obj["jc"][()].astype(int)),
                shape=(int(obj.attrs["MATLAB_sparse"]), len(obj["jc"]) - 1),
            )
        return {k: read_h5(h, v) for k, v in obj.items()}
    a = obj[()]
    if a.dtype.names and "real" in a.dtype.names:
        a = a["real"] + 1j * a["imag"]
    return a.T


def variables(path, names):
    if h5py.is_hdf5(path):
        with h5py.File(path) as h:
            return {k: read_h5(h, h[k]) for k in names}
    return loadmat(path, variable_names=names)


CASES = [
    dict(
        case="1",
        method="FEMONet",
        folder="A-proceeding",
        stem="A1_all",
        pred="ALL",
        scale=1,
        expected=4.39e-5,
    ),
    dict(
        case="2",
        method="FEMONet",
        folder="zy/B-proceeding",
        stem="B1_50688",
        pred="50688_ddp_save",
        scale=1,
        expected=1.35e-3,
    ),
    dict(
        case="3",
        method="FEMONet",
        folder="zy/C_proceeding",
        stem="C_3456",
        pred="3456",
        scale=1,
        expected=1.57e-5,
    ),
    dict(
        case="4",
        method="FEMONet",
        folder="SPP14",
        stem="SPP14_41",
        pred="41",
        scale=1,
        expected=1.70e-4,
    ),
    dict(
        case="5",
        method="FEMONet",
        folder="zy/261_data",
        stem="3Dcase3_261",
        pred="261_ddp_fft",
        scale=0.1,
        expected=1.33e-5,
    ),
    dict(
        case="SPP1",
        method="FEMONet",
        folder="SPP1",
        stem="SPP1_71",
        pred="71",
        scale=1,
        expected=None,
    ),
    dict(
        case="1",
        method="Enhanced VPINN",
        folder="A-proceeding",
        stem="A1_all",
        pred="unused",
        scale=1,
        expected=5.29e-4,
        pred_template="yh/plot/E_{split}_pred_deeponet_vpinn_5.mat",
    ),
    dict(
        case="OOD",
        method="FEMONet",
        folder="A-proceeding",
        stem="A6",
        pred="96",
        scale=1,
        expected=1.4e-3,
    ),
]


def summarize(rows):
    err = np.array([r["error_sq"] for r in rows])
    ref = np.array([r["reference_sq"] for r in rows])
    rel = np.sqrt(err / ref)
    mse = err.sum() / sum(r["dofs"] for r in rows)
    return dict(
        n_samples=len(rows),
        n_dofs=sum(r["dofs"] for r in rows),
        mse=float(mse),
        rmse=float(np.sqrt(mse)),
        absolute_l2_pooled=float(np.sqrt(err.sum())),
        absolute_l2_mean=float(np.sqrt(err).mean()),
        relative_l2_pooled=float(np.sqrt(err.sum() / ref.sum())),
        relative_l2_mean=float(rel.mean()),
        relative_l2_std=float(rel.std()),
        relative_l2_median=float(np.median(rel)),
        relative_l2_max=float(rel.max()),
    )


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    allrows = []
    summary = []
    for cfg in CASES:
        d = ROOT / cfg["folder"]
        data = variables(
            d / f"deepOnet_data_{cfg['stem']}.mat",
            ["Ez_test", "Ez_train", "coord_len_test", "coord_len_train"],
        )
        idx = variables(d / f"idx_{cfg['stem']}.mat", ["testIdx", "trainIdx"])
        for split in ["test", "train"]:
            path = d / f"E_{split}_pred_size_{cfg['pred']}.mat"
            if "pred_template" in cfg:
                path = ROOT / cfg["pred_template"].format(split=split)
            if not path.exists():
                print("MISSING", path, flush=True)
                continue
            pred = variables(path, ["E_pred"])["E_pred"]
            truth = data["Ez_" + split]
            lengths = data["coord_len_" + split].ravel().astype(int)
            ids = idx[split + "Idx"].ravel().astype(int)
            if cfg["method"] == "Enhanced VPINN":
                assert np.array_equal(
                    variables(path, ["sample_index"])["sample_index"].ravel(), ids
                )
            assert pred.shape == truth.shape and len(ids) == len(pred)
            rows = []
            for i, L in enumerate(lengths):
                p = pred[i, :L] * cfg["scale"]
                t = truth[i, :L] * cfg["scale"]
                e = p - t
                err = float(np.vdot(e, e).real)
                ref = float(np.vdot(t, t).real)
                assert ref > 0 and np.isfinite(err)
                rows.append(
                    dict(
                        case=cfg["case"],
                        method=cfg["method"],
                        split=split,
                        split_index=i + 1,
                        sample_id=int(ids[i]),
                        dofs=int(L),
                        error_sq=err,
                        reference_sq=ref,
                        absolute_l2=np.sqrt(err),
                        relative_l2=np.sqrt(err / ref),
                        mse=err / L,
                    )
                )
            s = dict(
                case=cfg["case"],
                method=cfg["method"],
                split=split,
                **summarize(rows),
                paper_mse=cfg["expected"] if split == "test" else None,
                prediction_path=str(path),
                data_path=str(d / f"deepOnet_data_{cfg['stem']}.mat"),
            )
            summary.append(s)
            allrows += rows
            print(json.dumps(s), flush=True)
            write_csv(OUT / "coefficient_summary.csv", summary)
            write_csv(OUT / "coefficient_per_sample.csv", allrows)
    (OUT / "coefficient_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
