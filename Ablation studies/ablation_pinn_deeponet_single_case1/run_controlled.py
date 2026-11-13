"""Run controlled."""

import argparse, csv, hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.io import savemat
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
REMOTE_BASE = ROOT.parent / "ablation_pinn_case1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, choices=[1000, 3000, 5000], required=True)
    ap.add_argument("--lbfgs", type=int, default=0)
    ap.add_argument("--baseline", type=Path, default=Path(REMOTE_BASE))
    ap.add_argument("--source", type=Path, default=(ROOT.parent / "A-proceeding"))
    args = ap.parse_args()
    sys.path.insert(0, str(args.baseline))
    import run_pinn as archived
    from reference_utils import sha256

    original = archived.original
    old = json.loads((args.baseline / "runs/pinn_seed123/manifest.json").read_text())
    assert sha256(args.baseline / "run_pinn.py") == old["runner_sha256"]
    assert (
        sha256(args.baseline / "source_reference/cnn_branch_test2.py")
        == old["source_sha256"]
    )
    assert sha256(args.baseline / "collocation.npz") == old["collocation_sha256"]
    assert sha256(args.source / "deepOnet_data_A1_all.mat") == old["data_sha256"]
    name = f"adam{args.epochs}" + (f"_lbfgs{args.lbfgs}" if args.lbfgs else "")
    out = ROOT / "runs" / name
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)
    checks = archived.derivative_check()
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    assert torch.cuda.is_available()
    cfg = original.PINNConfig(
        matpath=str(args.source / "deepOnet_data_A1_all.mat"),
        epochs=args.epochs,
        save_file_name="model_final",
    )
    model = original.DeepONet(1, 2, 128, 128).double()
    statehash = hashlib.sha256(
        b"".join(
            v.detach().cpu().numpy().tobytes() for v in model.state_dict().values()
        )
    ).hexdigest()
    assert statehash == old["initial_state_sha256"]
    pinn = archived.PINN(model, cfg)
    pinn.configure(1.0, 4)
    mesh = np.load(ROOT / "instance.npz")
    row = int(mesh["train_row"]) - 1
    assert row == 448 and int(mesh["sample_id"]) == 1
    dataset = pinn.train_set
    n = int(dataset.coord_len[row].item())
    np.testing.assert_array_equal(dataset.coord[row, :n].numpy(), mesh["vertices"])
    np.testing.assert_array_equal(dataset.Ez[row, :n].numpy(), mesh["truth"])
    trainloader = DataLoader(
        Subset(dataset, [row]), batch_size=cfg.batch_size, shuffle=True
    )
    batch = next(iter(trainloader))
    eps = batch[1]
    ids = batch[0].to(pinn.device)
    coords = batch[2].to(pinn.device)
    # Direct check of actual DeepONet coordinate Laplacian, not a surrogate network.
    model.eval()
    x = torch.tensor(
        [[[0.31, 0.24], [-0.71, 0.16], [0.47, -0.61]]],
        dtype=torch.float64,
        device=pinn.device,
        requires_grad=True,
    )
    re, im = pinn.E_function(eps, x)
    lap, _ = archived.differential(re, im, x)
    h = 1e-4
    with torch.no_grad():
        finite = torch.zeros_like(lap)
        base = torch.complex(re, im)
        for axis in (0, 1):
            shift = torch.zeros_like(x)
            shift[..., axis] = h
            a, b = pinn.E_function(eps, x + shift)
            c, d = pinn.E_function(eps, x - shift)
            finite += (torch.complex(a, b) - 2 * base + torch.complex(c, d)) / (h * h)
    torch.testing.assert_close(lap, finite, rtol=2e-4, atol=2e-5)
    checks["actual_deeponet_laplacian_finite_difference"] = "passed"
    del lap, finite, re, im, x
    # Independent spatial audit only: four random points per original triangle/edge.
    rng = np.random.default_rng(9101)
    tri = mesh["vertices"][mesh["triangles"]]
    q = rng.random((len(tri), 4, 2))
    s = np.sqrt(q[..., 0])
    bary = np.stack([1 - s, s * (1 - q[..., 1]), s * q[..., 1]], -1)
    xy = np.einsum("tqi,tij->tqj", bary, tri).reshape(-1, 2)
    er = np.repeat(np.where(mesh["domain"] == 1, 1.0, 1.45**2), 4)
    edges = np.sort(
        np.concatenate(
            [
                mesh["triangles"][:, [0, 1]],
                mesh["triangles"][:, [1, 2]],
                mesh["triangles"][:, [2, 0]],
            ]
        ),
        axis=1,
    )
    edges, cnt = np.unique(edges, axis=0, return_counts=True)
    edges = edges[cnt == 1]
    corners = mesh["vertices"][edges]
    tt = rng.random((len(edges), 4, 1))
    bb = ((1 - tt) * corners[:, :1] + tt * corners[:, 1:]).reshape(-1, 2)
    mid = corners.mean(1)
    norm = np.repeat(
        np.where(np.isclose(abs(mid), 1, atol=1e-12), np.sign(mid), 0.0), 4, axis=0
    )

    def dt(a):
        return torch.as_tensor(a, dtype=torch.float64, device=pinn.device)

    pinn.points["audit"] = {
        "xy": dt(xy)[None],
        "eps": dt(er)[None],
        "boundary": dt(bb)[None],
        "normal": dt(norm)[None],
        "interior_len": torch.tensor([len(xy)], device=pinn.device),
        "boundary_len": torch.tensor([len(bb)], device=pinn.device),
    }
    manifest = {
        "run": name,
        "reference_job": "28093",
        "sample_id": 1,
        "original_train_row": 449,
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "initial_state_sha256": statehash,
        "baseline_runner_sha256": old["runner_sha256"],
        "source_sha256": old["source_sha256"],
        "data_sha256": old["data_sha256"],
        "collocation_sha256": old["collocation_sha256"],
        "runner_sha256": sha256(Path(__file__)),
        "instance_sha256": sha256(ROOT / "instance.npz"),
        "adam_epochs": args.epochs,
        "adam_updates": args.epochs,
        "configured_batch_size": 32,
        "effective_batch_size": 1,
        "full_dataset_reference_updates": 39000,
        "full_dataset_reference_epochs": 1000,
        "scheduler": "original CosineAnnealingLR, T_max=specified Adam epochs, eta_min=1e-5; horizon extended for longer arms",
        "loss": "exact imported 28093 PINN.residual_sums; original fixed centroids and boundary midpoints; lambda_b=1; no scaling",
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "lbfgs_max_iterations": args.lbfgs,
        "selection": "predeclared endpoints only, no field-label or audit-based checkpoint selection",
        "checks": checks,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "job": os.getenv("SLURM_JOB_ID"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    torch.cuda.synchronize()
    start = time.monotonic()
    optimization_seconds = 0.0
    evaluations = 0
    f = (out / "loss.csv").open("w", newline="")
    writer = csv.writer(f)
    writer.writerow(
        [
            "stage",
            "step",
            "lr",
            "training_domain",
            "training_boundary",
            "training_total",
            "independent_domain",
            "independent_boundary",
            "mse",
            "relative_l2_nodes",
            "seconds",
        ]
    )

    def objective():
        a, b = pinn.residual_sums(eps, ids, "train")
        return (
            a / pinn.points["train"]["interior_len"][ids].sum()
            + b / pinn.points["train"]["boundary_len"][ids].sum()
        )

    def record(stage, step, export=False):
        model.eval()
        train = pinn.batch_loss(batch, "train")
        a, b = pinn.residual_sums(eps, torch.tensor([0], device=pinn.device), "audit")
        audit = (float(a.detach() / len(xy)), float(b.detach() / len(bb)))
        with torch.no_grad():
            re, im = pinn.E_function(eps, coords)
            p = torch.complex(re, im).cpu().numpy()[0, :n]
        error = p - mesh["truth"]
        mse = float(np.mean(abs(error) ** 2))
        rel = float(np.linalg.norm(error) / np.linalg.norm(mesh["truth"]))
        writer.writerow(
            [
                stage,
                step,
                pinn.optimizer.param_groups[0]["lr"],
                *train,
                sum(train),
                *audit,
                mse,
                rel,
                time.monotonic() - start,
            ]
        )
        f.flush()
        print(
            f"{stage} {step}: loss={sum(train):.8g} independent={sum(audit):.8g} MSE={mse:.8g} relL2={rel:.5g}",
            flush=True,
        )
        result = {
            "mse": mse,
            "relative_l2_nodes": rel,
            "training_domain": train[0],
            "training_boundary": train[1],
            "independent_domain": audit[0],
            "independent_boundary": audit[1],
            "complex_correlation": float(
                abs(np.vdot(mesh["truth"], p))
                / (np.linalg.norm(mesh["truth"]) * np.linalg.norm(p))
            ),
        }
        if export:
            torch.save(model.state_dict(), out / f"{stage}.pth")
            savemat(
                out / f"prediction_{stage}.mat",
                {
                    "prediction": p,
                    "truth": mesh["truth"],
                    "vertices": mesh["vertices"],
                    "triangles": mesh["triangles"] + 1,
                },
            )
        model.train()
        return result

    record("initial", 0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Exactly one original-size DataLoader batch, containing the sole instance.
        batch = next(iter(trainloader))
        torch.cuda.synchronize()
        tick = time.monotonic()
        pinn.optimizer.zero_grad(set_to_none=True)
        pinn.batch_loss(batch, "train", backward=True)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True
        )
        pinn.optimizer.step()
        pinn.scheduler.step()
        torch.cuda.synchronize()
        optimization_seconds += time.monotonic() - tick
        if epoch % 100 == 0:
            record("Adam", epoch)
    adam = record("adam_final", args.epochs, True)
    if args.lbfgs:
        model.train()
        opt = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=args.lbfgs,
            history_size=50,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
        )

        def closure():
            nonlocal evaluations
            opt.zero_grad(set_to_none=True)
            loss = objective()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite LBFGS loss")
            loss.backward()
            evaluations += 1
            if evaluations % 50 == 0:
                print(
                    f"LBFGS closure {evaluations}: loss={float(loss.detach()):.8g}",
                    flush=True,
                )
            return loss

        torch.cuda.synchronize()
        tick = time.monotonic()
        opt.step(closure)
        torch.cuda.synchronize()
        optimization_seconds += time.monotonic() - tick
        lbfgs_iterations = opt.state[next(iter(model.parameters()))].get("n_iter", 0)
    else:
        lbfgs_iterations = 0
    final = record("final", args.epochs, True)
    metrics = {
        "adam_final": adam,
        "final": final,
        "adam_updates": args.epochs,
        "lbfgs_iterations": lbfgs_iterations,
        "lbfgs_function_evaluations": evaluations,
        "optimization_seconds": optimization_seconds,
        "training_seconds_including_audits": time.monotonic() - start,
        "reference_full_dataset": {
            "epochs": 1000,
            "adam_updates": 39000,
            "training_seconds": 30873.54103770107,
            "test_mse": 0.9018013531588408,
            "note": "Aggregate test MSE is not this single training-instance error",
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    f.close()
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
