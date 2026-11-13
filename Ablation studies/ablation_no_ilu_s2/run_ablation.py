"""Run ablation."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source_reference"))
import numpy as np
import torch
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
import cnn_branch_test1 as original


class RawFEM(original.PINN_maxwell):
    def _build_ilu_cache(self):
        # No factorization, matrix scaling, RHS normalization, or ILU solve.
        pass

    def get_fem_loss(
        self, indices, epsilon_data, coord_data, E_true, Ai, Aj, Av, b, coord_len
    ):
        Ere_pred, Eim_pred = self.E_function(epsilon_data, coord_data)
        E = torch.complex(Ere_pred, Eim_pred)
        B, Mmax = E.shape
        Mi = coord_len.squeeze(-1).long().to(self.device)
        mask_x = torch.arange(Mmax, device=self.device)[None, :] < Mi[:, None]
        x_flat = E[mask_x]
        b_flat = b.to(self.device)[mask_x].to(x_flat.dtype)
        offsets = torch.cumsum(
            torch.cat([torch.zeros(1, device=self.device, dtype=torch.long), Mi[:-1]]),
            dim=0,
        )
        Ai, Aj = Ai.to(self.device).long(), Aj.to(self.device).long()
        Av = Av.to(self.device).to(x_flat.dtype)
        mask_nnz = (Ai > 0) & (Aj > 0)
        rows = (Ai - 1 + offsets.unsqueeze(1)).masked_select(mask_nnz)
        cols = (Aj - 1 + offsets.unsqueeze(1)).masked_select(mask_nnz)
        vals = Av.masked_select(mask_nnz)
        y = torch.zeros(int(Mi.sum().item()), dtype=x_flat.dtype, device=self.device)
        y.scatter_add_(0, rows, vals * x_flat.index_select(0, cols))
        r = y - b_flat
        return (r.abs() ** 2).mean()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def evaluate(pinn, split, out):
    dataset = getattr(pinn, split + "_set")
    pinn.model.eval()
    predictions, errors, lengths = [], [], []
    for _, eps, coords, truth, _, _, _, _, n in DataLoader(
        dataset, batch_size=32, shuffle=False
    ):
        re, im = pinn.E_function(eps, coords)
        p = torch.complex(re, im).cpu().numpy()
        y = truth.numpy()
        n = n.numpy().reshape(-1)
        mask = np.arange(p.shape[1])[None, :] < n[:, None]
        se = np.where(mask, np.abs(p - y) ** 2, 0).sum(axis=1)
        predictions.append(p)
        errors.extend(se.tolist())
        lengths.extend(n.tolist())
    errors, lengths = np.asarray(errors), np.asarray(lengths)
    mse = float(errors.sum() / lengths.sum())
    if not np.isfinite(mse):
        raise FloatingPointError("Non-finite field MSE")
    savemat(
        str(out / ("E_" + split + "_pred.mat")),
        {
            "E_pred": np.concatenate(predictions),
            "sample_mse": errors / lengths,
            "coord_len": lengths,
            "mse_global": mse,
        },
    )
    return {
        "mse_global": mse,
        "mse_sample_mean": float(np.mean(errors / lengths)),
        "samples": len(lengths),
        "valid_dofs": int(lengths.sum()),
    }


def check_raw_gradient():
    # Variable-length complex systems, including zero padding and duplicate COO entries.
    stub = RawFEM.__new__(RawFEM)
    stub.device = torch.device("cpu")
    x = torch.tensor(
        [[0.2 + 0.3j, -0.4 + 0.7j, 99j], [0.1 - 0.2j, 88j, 77j]],
        dtype=torch.complex128,
        requires_grad=True,
    )
    stub.E_function = lambda *args: (x.real, x.imag)
    ai = torch.tensor([[1, 1, 2, 2, 0], [1, 1, 0, 0, 0]])
    aj = torch.tensor([[1, 2, 1, 2, 0], [1, 1, 0, 0, 0]])
    av = torch.tensor(
        [[2 + 1j, 3, -1j, 4, 0], [1j, 2, 0, 0, 0]], dtype=torch.complex128
    )
    b = torch.tensor([[1j, 0.5, 100], [0.3j, 200, 300]], dtype=torch.complex128)
    loss = stub.get_fem_loss(
        None, None, None, None, ai, aj, av, b, torch.tensor([[2], [1]])
    )
    A = torch.tensor([[2 + 1j, 3], [-1j, 4]], dtype=torch.complex128)
    ref = (
        torch.abs(A @ x[0, :2] - b[0, :2]).square().sum()
        + torch.abs((2 + 1j) * x[1, 0] - b[1, 0]).square()
    ) / 3
    torch.testing.assert_close(loss, ref)
    g = torch.autograd.grad(loss, x, retain_graph=True)[0]
    gref = torch.autograd.grad(ref, x)[0]
    torch.testing.assert_close(g, gref)
    assert torch.count_nonzero(g[0, 2:]) == 0 and torch.count_nonzero(g[1, 1:]) == 0
    return {"loss": float(loss), "dense_reference_and_gradient": "passed"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["verify", "raw", "ilu"], required=True)
    ap.add_argument("--source", type=Path, default=(ROOT.parent / "SPP1"))
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args()
    if Path(args.run_name).name != args.run_name:
        raise ValueError("run-name must be a simple directory name")
    out = ROOT / "runs" / args.run_name
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite an existing run.
    os.chdir(out)
    (out / "model_save").mkdir()
    (out / "results").mkdir()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    source = args.source.resolve()
    cfg = original.PINNConfig(
        matpath=str(source / "deepOnet_data_SPP1_71.mat"),
        save_file_name="model_final",
        load_file_name="unused",
    )
    model = original.DeepONet(2, 4, 128, 64).double()
    initial_hash = hashlib.sha256(
        b"".join(
            v.detach().cpu().numpy().tobytes() for v in model.state_dict().values()
        )
    ).hexdigest()
    klass = original.PINN_maxwell if args.mode == "ilu" else RawFEM
    pinn = klass(model, cfg)
    assert len(pinn.train_set) == 57 and len(pinn.test_set) == 14
    manifest = {
        "mode": args.mode,
        "seed": args.seed,
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "data_sha256": sha256(source / "deepOnet_data_SPP1_71.mat"),
        "source_sha256": sha256(ROOT / "source_reference/cnn_branch_test1.py"),
        "loader_sha256": sha256(ROOT / "source_reference/getdata.py"),
        "initial_state_sha256": initial_hash,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "metric": "sum |E_pred-E_true|^2 / sum valid DOFs; complex field; no padding",
        "paper_test_mse": 3.7e-2,
        "historical_initial_seed": "not recorded; new paired runs use identical seed and initial weights",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.mode == "verify":
        checks = check_raw_gradient()
        checkpoint_audit = {}
        for checkpoint in (
            source / "model_SPP1_size_71",
            source / "model_save/model_SPP1_size_71_epoch1000.pth",
        ):
            if not checkpoint.exists():
                continue
            pinn.model.load_state_dict(
                torch.load(checkpoint, map_location=pinn.device, weights_only=True),
                strict=True,
            )
            checkpoint_audit[str(checkpoint)] = {
                s: evaluate(pinn, s, out) for s in ("train", "test")
            }
            checkpoint_audit[str(checkpoint)]["sha256"] = sha256(checkpoint)
        result = {}
        for split in ("train", "test"):
            pred = loadmat(
                ROOT / "reference_predictions" / f"E_{split}_pred_size_71.mat"
            )["E_pred"]
            dataset = getattr(pinn, split + "_set")
            truth = dataset.Ez.numpy()
            lengths = dataset.coord_len.numpy().reshape(-1)
            mask = np.arange(pred.shape[1])[None, :] < lengths[:, None]
            se = np.where(mask, np.abs(pred - truth) ** 2, 0).sum(axis=1)
            mse = float(se.sum() / lengths.sum())
            result[split] = dict(
                mse_global=mse,
                mse_sample_mean=float(np.mean(se / lengths)),
                samples=len(lengths),
                valid_dofs=int(lengths.sum()),
            )
            savemat(
                out / f"E_{split}_pred.mat",
                dict(
                    E_pred=pred,
                    sample_mse=se / lengths,
                    coord_len=lengths,
                    mse_global=mse,
                ),
            )
        assert abs(result["test"]["mse_global"] / 0.03733687180069727 - 1) < 1e-8
        checks["checkpoint_audit"] = checkpoint_audit
        checks["reference_prediction_sha256"] = {
            s: sha256(ROOT / "reference_predictions" / f"E_{s}_pred_size_71.mat")
            for s in ("train", "test")
        }
        checks["reference_source"] = (
            "Local paper-matching SPP1 predictions; remote predictions belong to a different run (test MSE 0.0442326331374)."
        )
        checks["archived_predictions_match_reference"] = True
        checks["checkpoint_reproduces_archived_mse"] = any(
            abs(v["test"]["mse_global"] / result["test"]["mse_global"] - 1) < 1e-5
            for v in checkpoint_audit.values()
        )
        checks["note"] = (
            "Historical predictions define Fig. S2 reference; current checkpoints audited separately. New matched runs use the frozen available source."
        )
        result["checks"] = checks
    else:
        # Require successful checkpoint/data/gradient verification before long training.
        verification = json.loads((ROOT / "runs/verify/metrics.json").read_text())
        assert verification["checks"]["archived_predictions_match_reference"]
        verified_manifest = json.loads((ROOT / "runs/verify/manifest.json").read_text())
        for key in ("data_sha256", "source_sha256", "loader_sha256"):
            assert manifest[key] == verified_manifest[key]
        start = time.monotonic()
        pinn.train(cfg.epochs, print_every=cfg.print_every, save_every=cfg.save_every)
        elapsed = time.monotonic() - start
        with (out / "loss.csv").open("w", newline="") as f:
            csv.writer(f).writerows(pinn.losses)
        result = {s: evaluate(pinn, s, out) for s in ("train", "test")}
        result["training_seconds"] = elapsed
        result["test_mse_ratio_to_archived_ilu"] = (
            result["test"]["mse_global"] / verification["test"]["mse_global"]
        )
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
