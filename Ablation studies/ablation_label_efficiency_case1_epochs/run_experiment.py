"""Run experiment."""

import argparse, copy, csv, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import default_collate
from reference_utils import original, evaluate, sha256, RawFEM

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "A-proceeding"
COUNTS = [10, 30, 100, 300, 1000]
CONFIG = dict(
    epochs=1000,
    protocol="one_pass_subset_per_epoch",
    batch_size=32,
    lr=1e-3,
    eta_min=1e-5,
    clip=1.0,
    dtype="float64/complex128",
    hybrid_weight=1.0,
)


def dump(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def state_hash(model):
    return hashlib.sha256(
        b"".join(
            v.detach().cpu().numpy().tobytes() for v in model.state_dict().values()
        )
    ).hexdigest()


def initialize(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return original.DeepONet(1, 2, 128, 128).double()


def labels(seed, n):
    return np.random.default_rng(seed + 1000).permutation(1246)[:n]


def batch(dataset, ids):
    return default_collate([dataset[int(i)] for i in ids])


def epoch_batches(mode, n, seed, rng):
    pool = np.arange(1246) if mode == "pi" else labels(seed, n)
    order = rng.permutation(pool)
    return [order[i : i + 32].tolist() for i in range(0, len(order), 32)]


def masked_mse(pred, truth, lengths):
    mask = torch.arange(pred.shape[1], device=pred.device)[None, :] < lengths.to(
        pred.device
    ).reshape(-1, 1)
    return (pred[mask] - truth.to(pred.device)[mask]).abs().square().mean()


def supervised(p, ids):
    d = p.train_set
    re, im = p.E_function(d.epsilon[ids], d.coord[ids])
    return masked_mse(torch.complex(re, im), d.Ez[ids], d.coord_len[ids])


def objective(p, mode, physical, supervised_ids, weight=1.0):
    zero = torch.zeros((), device=p.device, dtype=torch.float64)
    lp = p.get_fem_loss(*physical) if mode != "dd" else zero
    ld = supervised(p, supervised_ids) if mode != "pi" and weight else zero
    return lp + weight * ld, lp, ld


def step(p, mode, physical, ids, weight=1.0):
    p.model.train()
    p.optimizer.zero_grad()
    loss, lp, ld = objective(p, mode, physical, ids, weight)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(p.model.parameters(), 1.0, error_if_nonfinite=True)
    p.optimizer.step()
    return [float(x.detach()) for x in (loss, lp, ld)]


def verify(p, out, seed):
    x = torch.tensor(
        [[1 + 2j, 3j, 999j], [2j, 888j, 777j]],
        dtype=torch.complex128,
        requires_grad=True,
    )
    y = torch.zeros_like(x)
    n = torch.tensor([2, 1])
    loss = masked_mse(x, y, n)
    ref = (x[0, :2].abs().square().sum() + x[1, 0].abs().square()) / 3
    torch.testing.assert_close(loss, ref)
    g = torch.autograd.grad(loss, x)[0]
    assert g[0, 2] == 0 and torch.count_nonzero(g[1, 1:]) == 0
    torch.testing.assert_close(g[0, :2], 2 * x.detach()[0, :2] / 3)
    split = loadmat(ROOT / "source_reference/idx_A1_all.mat")
    a, b = split["trainIdx"].ravel(), split["testIdx"].ravel()
    assert len(a) == 1246 and len(b) == 312 and len(set(a) & set(b)) == 0
    assert len(set(a) | set(b)) == 1558
    for n in COUNTS:
        assert set(labels(seed, n)) <= set(labels(seed, 1000))
    initial = copy.deepcopy(p.model.state_dict())
    ids = labels(seed, 10).tolist()
    pb = batch(p.train_set, range(2))

    def gradients(mode, weight=1.0):
        p.model.load_state_dict(initial)
        p.model.train()
        p.optimizer.zero_grad()
        value = objective(p, mode, pb, ids, weight)[0]
        value.backward()
        return value.detach().clone(), [
            v.grad.detach().clone() for v in p.model.parameters()
        ]

    for mode in ("pi", "dd", "hybrid"):
        before = gradients(mode)
        truth = p.train_set.Ez.clone()
        forbidden = np.setdiff1d(np.arange(1246), ids)
        p.train_set.Ez[forbidden] = complex(float("nan"), float("nan"))
        # Physics receives poisoned labels too: the residual must ignore them.
        saved = pb[3]
        pb[3] = torch.full_like(saved, complex(float("nan"), 0))
        after = gradients(mode)
        p.train_set.Ez.copy_(truth)
        pb[3] = saved
        torch.testing.assert_close(before[0], after[0])
        for u, v in zip(before[1], after[1]):
            torch.testing.assert_close(u, v)
    u, v = gradients("pi"), gradients("hybrid", 0)
    for aa, bb in zip(u[1], v[1]):
        torch.testing.assert_close(aa, bb)
    states = []
    for mode in ("pi", "hybrid"):
        p.model.load_state_dict(initial)
        p.optimizer = torch.optim.Adam(p.model.parameters(), lr=1e-3)
        step(p, mode, pb, ids, weight=0.0 if mode == "hybrid" else 1.0)
        states.append(copy.deepcopy(p.model.state_dict()))
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key])
    # Validate scheduler boundaries and the exact update budget without expensive forwards.
    parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    opt = torch.optim.Adam([parameter], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=1000, eta_min=1e-5
    )
    rates = []
    for cycle in range(1000):
        rates.append(opt.param_groups[0]["lr"])
        opt.step()
        scheduler.step()
    assert len(rates) == 1000 and rates[0] == 1e-3
    for count in COUNTS:
        dd = epoch_batches("dd", count, seed, np.random.default_rng(seed + 3000))
        hybrid = epoch_batches(
            "hybrid", count, seed, np.random.default_rng(seed + 3000)
        )
        flat = [i for chunk in dd for i in chunk]
        assert dd == hybrid and len(flat) == count and len(set(flat)) == count
        assert set(flat) == set(labels(seed, count))
        assert len(dd) == (count + 31) // 32
    assert abs(opt.param_groups[0]["lr"] - 1e-5) < 1e-15
    assert state_hash(initialize(seed)) == state_hash(initialize(seed))
    smoke = {}
    for mode in ("pi", "dd", "hybrid"):
        p.model.load_state_dict(initial)
        p.optimizer = torch.optim.Adam(p.model.parameters(), lr=1e-3)
        start = state_hash(p.model)
        smoke_batch = batch(p.train_set, ids) if mode == "hybrid" else pb
        smoke[mode] = [step(p, mode, smoke_batch, ids) for _ in range(2)]
        assert state_hash(p.model) != start
    p.model.load_state_dict(
        torch.load(
            SOURCE / "model_A1_size_ALL", map_location=p.device, weights_only=True
        ),
        strict=True,
    )
    metrics = evaluate(p, "test", out)
    assert abs(metrics["mse_global"] / 4.390900391795585e-5 - 1) < 1e-5
    return dict(
        passed=True,
        archived_test=metrics,
        smoke=smoke,
        checks=[
            "complex_mask_gradient",
            "split_disjoint",
            "nested_labels",
            "unauthorized_label_invariance",
            "hybrid_zero_weight_gradient_and_update",
            "epoch_exactly_once_and_matched_hybrid_subset",
            "repeatable_initialization",
            "three_mode_smoke",
        ],
    )


def main():
    global SOURCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--mode", choices=["verify", "pi", "dd", "hybrid"], required=True)
    ap.add_argument("--n-label", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args()
    SOURCE = args.source.resolve()
    assert Path(args.run_name).name == args.run_name
    assert (args.mode in ("verify", "pi") and args.n_label == 0) or (
        args.mode in ("dd", "hybrid") and args.n_label in COUNTS
    )
    out = ROOT / "runs" / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)
    assert torch.cuda.is_available(), "A CUDA GPU is required"
    model = initialize(args.seed)
    manifest = dict(
        mode=args.mode,
        n_label=args.n_label,
        seed=args.seed,
        config=CONFIG,
        initial_state_sha256=state_hash(model),
        data_sha256=sha256(SOURCE / "deepOnet_data_A1_all.mat"),
        source_sha256=sha256(ROOT / "source_reference/cnn_branch_test2.py"),
        implementation_sha256=sha256(ROOT / "run_experiment.py"),
        loader_sha256=sha256(ROOT / "source_reference/getdata.py"),
        evaluator_sha256=sha256(ROOT / "reference_utils.py"),
        split_sha256=sha256(ROOT / "source_reference/idx_A1_all.mat"),
        labels=labels(args.seed, args.n_label).tolist(),
        slurm_job_id=os.getenv("SLURM_JOB_ID"),
        n_physics=(
            1246
            if args.mode in ("pi", "verify")
            else (args.n_label if args.mode == "hybrid" else 0)
        ),
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
    )
    dump(out / "manifest.json", manifest)
    if args.mode != "verify":
        assert json.loads((ROOT / "runs/verify/metrics.json").read_text())["passed"]
        vm = json.loads((ROOT / "runs/verify/manifest.json").read_text())
        for k in (
            "data_sha256",
            "source_sha256",
            "implementation_sha256",
            "initial_state_sha256",
            "loader_sha256",
            "evaluator_sha256",
            "split_sha256",
        ):
            assert vm[k] == manifest[k]
    start = time.monotonic()
    cls = RawFEM if args.mode == "dd" else original.PINN_maxwell
    p = cls(
        model, original.PINNConfig(matpath=str(SOURCE / "deepOnet_data_A1_all.mat"))
    )
    assert len(p.train_set) == 1246 and len(p.test_set) == 312
    setup_seconds = time.monotonic() - start
    if args.mode == "verify":
        dump(out / "metrics.json", verify(p, out, args.seed))
        return
    # Erase unauthorized training labels and ALL test labels until final evaluation.
    train_truth, test_truth = p.train_set.Ez.clone(), p.test_set.Ez.clone()
    forbidden = np.setdiff1d(np.arange(1246), manifest["labels"])
    p.train_set.Ez[forbidden] = complex(float("nan"), 0)
    p.test_set.Ez[:] = complex(float("nan"), 0)
    physical_rng = np.random.default_rng(args.seed + 3000)
    ph, lh = hashlib.sha256(), hashlib.sha256()
    start = time.monotonic()
    updates = 0
    with (out / "loss.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "updates", "lr", "loss", "physics", "supervised"])
        for cycle in range(1000):
            batches = epoch_batches(args.mode, args.n_label, args.seed, physical_rng)
            values = []
            lr = p.optimizer.param_groups[0]["lr"]
            for phys_ids in batches:
                physical = batch(p.train_set, phys_ids) if args.mode != "dd" else None
                sup_ids = phys_ids if args.mode != "pi" else None
                if physical is not None:
                    ph.update(np.asarray(phys_ids, dtype="<i8").tobytes())
                if sup_ids is not None:
                    lh.update(np.asarray(sup_ids, dtype="<i8").tobytes())
                values.append(step(p, args.mode, physical, sup_ids))
                updates += 1
            p.scheduler.step()
            row = [cycle + 1, updates, lr, *np.mean(values, axis=0).tolist()]
            writer.writerow(row)
            f.flush()
            print(row, flush=True)
    elapsed = time.monotonic() - start
    expected_updates = 1000 * (
        ((1246 if args.mode == "pi" else args.n_label) + 31) // 32
    )
    assert updates == expected_updates
    torch.save(p.model.state_dict(), out / "model_final.pth")
    p.train_set.Ez.copy_(train_truth)
    p.test_set.Ez.copy_(test_truth)
    metrics = {s: evaluate(p, s, out) for s in ("train", "test")}
    metrics.update(
        updates=updates,
        training_seconds=elapsed,
        setup_seconds=setup_seconds,
        physical_order_sha256=ph.hexdigest(),
        label_order_sha256=lh.hexdigest(),
        final_lr=p.optimizer.param_groups[0]["lr"],
    )
    dump(out / "metrics.json", metrics)


if __name__ == "__main__":
    main()
