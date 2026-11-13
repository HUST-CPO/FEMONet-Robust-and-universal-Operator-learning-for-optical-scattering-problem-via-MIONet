"""Reference utils."""

import hashlib
import numpy as np
import torch
from scipy.io import savemat
from torch.utils.data import DataLoader


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
