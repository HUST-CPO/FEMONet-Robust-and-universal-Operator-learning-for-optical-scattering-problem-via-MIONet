"""Run pinn."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source_reference"))
import cnn_branch_test2 as original
from reference_utils import evaluate, sha256

K = 2 * np.pi / 1.55


def grad(y, x):
    return torch.autograd.grad(
        y, x, torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


def differential(re, im, x):
    gr, gi = grad(re, x), grad(im, x)
    lr = sum(grad(gr[..., a], x)[..., a] for a in (0, 1))
    li = sum(grad(gi[..., a], x)[..., a] for a in (0, 1))
    return torch.complex(lr, li), torch.complex(gr, gi)


class PINN(original.PINN_maxwell):
    def _build_ilu_cache(self):
        pass

    def configure(self, weight, microbatch):
        self.boundary_weight = weight
        self.microbatch = microbatch
        z = np.load(ROOT / "collocation.npz")
        self.points = {
            s: {
                k: torch.as_tensor(
                    z[s + "_" + k],
                    device=self.device,
                    dtype=torch.long if k.endswith("_len") else torch.float64,
                )
                for k in (
                    "xy",
                    "eps",
                    "boundary",
                    "normal",
                    "interior_len",
                    "boundary_len",
                )
            }
            for s in ("train", "test")
        }

    def residual_sums(self, eps, ids, split):
        p = self.points[split]
        ni = p["interior_len"][ids]
        nb = p["boundary_len"][ids]
        mi, mb = int(ni.max()), int(nb.max())
        xy = p["xy"][ids, :mi].detach().requires_grad_(True)
        boundary = p["boundary"][ids, :mb].detach().requires_grad_(True)
        # Compute branch once; trunk remains pointwise and smooth (tanh).
        b = self.model.branch_net(
            original.add_spatial_coord_channels(eps.to(self.device))
        )

        def field(x):
            t = self.model.trunk_net(x)
            h = self.model.output_dim // 2
            return torch.einsum("bi,bni->bn", b[:, :h], t[:, :, :h]), torch.einsum(
                "bi,bni->bn", b[:, h:], t[:, :, h:]
            )

        re, im = field(xy)
        lap, _ = differential(re, im, xy)
        u = torch.complex(re, im)
        er = p["eps"][ids, :mi]
        r = -lap - K * K * er * u - K * K * (er - 1) * torch.exp(-1j * K * xy[..., 0])
        br, bi = field(boundary)
        normal = p["normal"][ids, :mb]
        rb = (torch.complex(grad(br, boundary), grad(bi, boundary)) * normal).sum(
            -1
        ) + 1j * K * torch.complex(br, bi)
        mask = torch.arange(mi, device=self.device)[None, :] < ni[:, None]
        bm = torch.arange(mb, device=self.device)[None, :] < nb[:, None]
        return r[mask].abs().square().sum(), rb[bm].abs().square().sum()

    def batch_loss(self, batch, split, backward=False):
        ids, eps = batch[:2]
        ids = ids.to(self.device) - (len(self.train_set) if split == "test" else 0)
        p = self.points[split]
        ni = p["interior_len"][ids].sum()
        nb = p["boundary_len"][ids].sum()
        domain_value = boundary_value = 0.0
        for start in range(0, len(ids), self.microbatch):
            sl = slice(start, start + self.microbatch)
            a, b = self.residual_sums(eps[sl], ids[sl], split)
            loss = a / ni + self.boundary_weight * b / nb
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PINN loss")
            if backward:
                loss.backward()
            domain_value += float(a.detach() / ni)
            boundary_value += float(b.detach() / nb)
        return domain_value, boundary_value


def derivative_check():
    torch.manual_seed(81)
    x = torch.randn(2, 7, 2, dtype=torch.float64, requires_grad=True)
    a = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    re = a * (x[..., 0] ** 3 + x[..., 1] ** 2)
    im = a * (x[..., 0] ** 2 * x[..., 1] + x[..., 1] ** 3)
    lap, g = differential(re, im, x)
    expected = a * torch.complex(6 * x[..., 0] + 2, 8 * x[..., 1])
    torch.testing.assert_close(lap, expected)
    analytical = a * torch.stack(
        [
            torch.complex(3 * x[..., 0] ** 2, 2 * x[..., 0] * x[..., 1]),
            torch.complex(2 * x[..., 1], x[..., 0] ** 2 + 3 * x[..., 1] ** 2),
        ],
        -1,
    )
    torch.testing.assert_close(g, analytical)
    lhs = (lap.abs() ** 2).sum()
    rhs = (expected.abs() ** 2).sum()
    torch.testing.assert_close(
        torch.autograd.grad(lhs, a, retain_graph=True)[0],
        torch.autograd.grad(rhs, a)[0],
    )
    # Outgoing plane wave obeys Robin on the +x face with this time convention.
    y = torch.randn(1, 9, 2, dtype=torch.float64, requires_grad=True)
    u = torch.exp(-1j * K * y[..., 0])
    dg = torch.complex(grad(u.real, y), grad(u.imag, y))
    torch.testing.assert_close(
        dg[..., 0] + 1j * K * u, torch.zeros_like(u), atol=1e-12, rtol=0
    )
    # u_s=-u_inc satisfies the inhomogeneous bulk equation for arbitrary eps.
    lap, _ = differential(-u.real, -u.imag, y)
    er = 2.1025
    residual = -lap + K * K * er * u - K * K * (er - 1) * u
    torch.testing.assert_close(residual, torch.zeros_like(u), atol=1e-12, rtol=0)
    return {
        "polynomial_laplacian_and_parameter_gradient": "passed",
        "outgoing_robin_sign": "passed",
        "scattered_source_sign": "passed",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["verify", "train"], required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--source", type=Path, default=(ROOT.parent / "A-proceeding"))
    ap.add_argument(
        "--reference", type=Path, default=(ROOT.parent / "ablation_no_ilu_case1")
    )
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--boundary-weight", type=float, default=1.0)
    ap.add_argument("--microbatch", type=int, default=4)
    args = ap.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("Invalid run name")
    assert args.microbatch > 0 and args.boundary_weight > 0
    out = ROOT / "runs" / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)
    (out / "model_save").mkdir()
    checks = derivative_check()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    cfg = original.PINNConfig(
        matpath=str(args.source / "deepOnet_data_A1_all.mat"),
        save_file_name="model_final",
    )
    model = original.DeepONet(1, 2, 128, 128).double()
    initial = hashlib.sha256(
        b"".join(
            v.detach().cpu().numpy().tobytes() for v in model.state_dict().values()
        )
    ).hexdigest()
    pinn = PINN(model, cfg)
    pinn.configure(args.boundary_weight, args.microbatch)
    assert (len(pinn.train_set), len(pinn.test_set)) == (1246, 312)
    audit = json.loads((ROOT / "collocation_audit.json").read_text())
    manifest = {
        "mode": args.mode,
        "seed": args.seed,
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "initial_state_sha256": initial,
        "data_sha256": sha256(args.source / "deepOnet_data_A1_all.mat"),
        "source_sha256": sha256(ROOT / "source_reference/cnn_branch_test2.py"),
        "runner_sha256": sha256(Path(__file__)),
        "collocation_sha256": sha256(ROOT / "collocation.npz"),
        "boundary_weight": args.boundary_weight,
        "microbatch": args.microbatch,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "loss": "mean(-laplacian(u)-k^2*eps*u-k^2*(eps-1)*exp(-i*k*x)) magnitude squared + lambda_b*mean(|dn(u)+i*k*u|^2)",
        "coordinate_unit": "micrometre; k=2*pi/1.55; no additional residual scaling",
        "reduction": "batch pooled valid interior points and boundary points separately",
        "collocation_audit": audit,
    }
    paired = json.loads((args.reference / "runs/ilu_seed123/manifest.json").read_text())
    for key in ("initial_state_sha256", "data_sha256", "source_sha256"):
        assert manifest[key] == paired[key], key
    assert manifest["data_sha256"] == audit["data_sha256"]
    assert manifest["collocation_sha256"] == audit["collocation_sha256"]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.mode == "verify":
        pinn.model.eval()
        saved = torch.load(
            args.source / "model_A1_size_ALL",
            map_location=pinn.device,
            weights_only=True,
        )
        pinn.model.load_state_dict(saved, strict=True)
        metrics = {s: evaluate(pinn, s, out) for s in ("train", "test")}
        assert abs(metrics["test"]["mse_global"] / 4.390900391795585e-5 - 1) < 1e-5
        # Prove chunked accumulation has the same loss and gradients as a full batch.
        batch = next(iter(DataLoader(pinn.train_set, batch_size=2, shuffle=False)))
        pin = []
        for size in (2, 1):
            pinn.microbatch = size
            pinn.optimizer.zero_grad(set_to_none=True)
            vals = pinn.batch_loss(batch, "train", backward=True)
            pin.append((vals, [p.grad.detach().clone() for p in model.parameters()]))
        np.testing.assert_allclose(pin[0][0], pin[1][0], rtol=1e-10)
        for g, h in zip(pin[0][1], pin[1][1]):
            torch.testing.assert_close(g, h, rtol=1e-8, atol=1e-9)
        pinn.optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        checks.update(
            archived_checkpoint_reproduces_table2=True,
            microbatch_loss_and_gradient="passed",
            optimizer_step="passed",
        )
        metrics["checks"] = checks
    else:
        verification = json.loads((ROOT / "runs/verify/metrics.json").read_text())
        assert verification["checks"]["archived_checkpoint_reproduces_table2"]
        vm = json.loads((ROOT / "runs/verify/manifest.json").read_text())
        for key in (
            "data_sha256",
            "source_sha256",
            "runner_sha256",
            "collocation_sha256",
            "boundary_weight",
        ):
            assert vm[key] == manifest[key]
        start = time.monotonic()
        with (out / "loss.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_total",
                    "train_domain",
                    "train_boundary",
                    "test_total",
                    "test_domain",
                    "test_boundary",
                    "lr",
                ]
            )
            for epoch in range(cfg.epochs):
                pinn.model.train()
                train = np.zeros(2)
                for batch in pinn.train_loader:
                    pinn.optimizer.zero_grad(set_to_none=True)
                    train += pinn.batch_loss(batch, "train", backward=True)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True
                    )
                    pinn.optimizer.step()
                train /= len(pinn.train_loader)
                pinn.model.eval()
                # One logical full validation batch, chunked with correct global denominators.
                test = np.array(pinn.batch_loss(next(iter(pinn.test_loader)), "test"))
                tr = train[0] + args.boundary_weight * train[1]
                te = test[0] + args.boundary_weight * test[1]
                writer.writerow(
                    [
                        epoch + 1,
                        tr,
                        *train,
                        te,
                        *test,
                        pinn.optimizer.param_groups[0]["lr"],
                    ]
                )
                f.flush()
                pinn.scheduler.step()
                print(
                    f"Epoch {epoch+1}: train={tr:.12g} test={te:.12g} domain={train[0]:.12g} boundary={train[1]:.12g}",
                    flush=True,
                )
                if (epoch + 1) % cfg.save_every == 0:
                    torch.save(
                        model.state_dict(),
                        out / "model_save" / f"model_epoch{epoch+1}.pth",
                    )
        torch.save(model.state_dict(), out / "model_final")
        elapsed = time.monotonic() - start
        metrics = {s: evaluate(pinn, s, out) for s in ("train", "test")}
        metrics["training_seconds"] = elapsed
        metrics["epochs"] = cfg.epochs
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
