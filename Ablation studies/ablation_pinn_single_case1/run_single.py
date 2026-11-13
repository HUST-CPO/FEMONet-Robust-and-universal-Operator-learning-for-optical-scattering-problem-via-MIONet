"""Run single."""

import argparse, copy, csv, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spilu, spsolve
from scipy.io import savemat

ROOT = Path(__file__).resolve().parent
K = 2 * np.pi / 1.55
DEVICE = "cuda"
torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)


def derivative(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def field(model, x, lap=True):
    z = model(x)
    u = torch.complex(z[:, 0], z[:, 1])
    g0 = derivative(z[:, 0], x)
    g1 = derivative(z[:, 1], x)
    g = torch.complex(g0, g1)
    if not lap:
        return u, g
    l0 = sum(derivative(g0[:, a], x)[:, a] for a in (0, 1))
    l1 = sum(derivative(g1[:, a], x)[:, a] for a in (0, 1))
    return u, torch.complex(l0, l1)


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float64, device=DEVICE)


class Problem:
    def __init__(self, manufactured=False):
        self.d = np.load(ROOT / "instance.npz")
        self.manufactured = manufactured
        self.v = self.d["vertices"]
        self.t = self.d["triangles"]
        self.p = self.v[self.t]
        self.area = abs(np.linalg.det(self.p[:, 1:] - self.p[:, :1])) / 2
        self.domain = self.d["domain"]
        self.er = np.where(self.domain == 1, 1.0, 1.45**2)
        edges = np.sort(
            np.concatenate([self.t[:, [0, 1]], self.t[:, [1, 2]], self.t[:, [2, 0]]]),
            axis=1,
        )
        edges, n = np.unique(edges, axis=0, return_counts=True)
        self.edges = edges[n == 1]
        self.bp = self.v[self.edges].mean(1)
        self.normal = np.where(
            np.isclose(abs(self.bp), 1, atol=1e-12), np.sign(self.bp), 0.0
        )
        assert np.all((self.normal**2).sum(1) == 1)

    def sample(self, n, nb, rng, fixed=False):
        if fixed:
            x = self.p.mean(1)
            eps = self.er
            b = self.bp
            normal = self.normal
        else:
            # Equal numbers from each material; area-uniform within each material.
            ii = []
            for domain in (1, 2):
                choices = np.flatnonzero(self.domain == domain)
                ii.extend(
                    rng.choice(
                        choices, n // 2, p=self.area[choices] / self.area[choices].sum()
                    )
                )
            ii = np.asarray(ii)
            q = rng.random((len(ii), 2))
            a = np.sqrt(q[:, 0])
            bary = np.stack([1 - a, a * (1 - q[:, 1]), a * q[:, 1]], 1)
            x = (self.p[ii] * bary[:, :, None]).sum(1)
            eps = self.er[ii]
            s = rng.uniform(-1, 1, nb)
            side = np.arange(nb) % 4
            b = np.stack([s, s], 1)
            normal = np.zeros_like(b)
            for a, (axis, sign) in enumerate(((0, -1), (0, 1), (1, -1), (1, 1))):
                mask = side == a
                b[mask, axis] = sign
                normal[mask, axis] = sign
        if self.manufactured:
            eps = np.ones_like(eps)
        return tuple(tensor(z) for z in (x, eps, b, normal))

    def residual(self, model, batch):
        xx, eps, bb, normal = batch
        x = xx.detach().requires_grad_(True)
        b = bb.detach().requires_grad_(True)
        u, lap = field(model, x)
        if self.manufactured:
            exact = torch.exp(1j * (x[:, 0] + 0.5 * x[:, 1]))
            source = (1.25 - K * K) * exact
        else:
            source = K * K * (eps - 1) * torch.exp(-1j * K * x[:, 0])
        r = -lap - K * K * eps * u - source
        ub, g = field(model, b, lap=False)
        rb = (g * normal).sum(1) + 1j * K * ub
        if self.manufactured:
            exact = torch.exp(1j * (b[:, 0] + 0.5 * b[:, 1]))
            rb = rb - 1j * (normal[:, 0] + 0.5 * normal[:, 1] + K) * exact
        return r.abs().square().mean(), rb.abs().square().mean()


def verify(model, problem):
    class Exact(nn.Module):
        def forward(self, x):
            phase = x[:, 0] + 0.5 * x[:, 1]
            return torch.stack([torch.cos(phase), torch.sin(phase)], 1)

    manufactured = Problem(True)
    a, b = manufactured.residual(
        Exact(), manufactured.sample(32, 32, np.random.default_rng(51))
    )
    assert float(a) < 1e-24 and float(b) < 1e-24
    x = tensor([[0.31, 0.24], [-0.71, 0.16], [0.47, -0.61]]).requires_grad_(True)
    u, lap = field(model, x)
    h = 1e-4
    with torch.no_grad():

        def value(y):
            z = model(y)
            return torch.complex(z[:, 0], z[:, 1])

        finite = torch.zeros_like(u)
        for a in (0, 1):
            shift = torch.zeros_like(x)
            shift[:, a] = h
            finite += (value(x + shift) - 2 * value(x) + value(x - shift)) / (h * h)
    torch.testing.assert_close(lap, finite, atol=2e-6, rtol=2e-4)
    # Check the complete strong loss's parameter gradient with central differences.
    tiny = problem.sample(8, 8, np.random.default_rng(91))

    def loss():
        a, b = problem.residual(model, tiny)
        return a / K**4 + b / K**2

    v = loss()
    parameter = list(model.parameters())[-1]
    gradient = torch.autograd.grad(v, parameter)[0][0].item()
    delta = 1e-6
    with torch.no_grad():
        parameter[0] += delta
    plus = loss().item()
    with torch.no_grad():
        parameter[0] -= 2 * delta
    minus = loss().item()
    with torch.no_grad():
        parameter[0] += delta
    np.testing.assert_allclose(
        gradient, (plus - minus) / (2 * delta), atol=1e-6, rtol=1e-5
    )
    return {
        "manufactured_exact_PDE_and_Robin": "passed",
        "network_laplacian_finite_difference": "passed",
        "loss_parameter_finite_difference": "passed",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["manufactured", "fixed", "resample1", "resample10", "ilu"],
        required=True,
    )
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lbfgs", type=int, default=500)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.tag and (not args.tag.replace("_", "").isalnum()):
        raise ValueError("Invalid tag")
    suffix = "_" + args.tag if args.tag else ""
    out = ROOT / "runs" / f"{args.mode}_seed{args.seed}{suffix}"
    out.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Standard coordinate-only PINN, also used unchanged by the ILU comparator.
    layers = [nn.Linear(2, 64), nn.Tanh()]
    for _ in range(5):
        layers.extend([nn.Linear(64, 64), nn.Tanh()])
    layers.append(nn.Linear(64, 2))
    model = nn.Sequential(*layers).to(DEVICE)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    initial = hashlib.sha256(
        b"".join(
            v.detach().cpu().numpy().tobytes() for v in model.state_dict().values()
        )
    ).hexdigest()
    problem = Problem(args.mode == "manufactured")
    checks = verify(model, problem)
    # Restore exact initialization after finite-difference perturbations.
    torch.manual_seed(args.seed)
    # Finite differences only round the perturbed bias; restore its known zero value.
    with torch.no_grad():
        list(model.parameters())[-1].zero_()
    assert (
        initial
        == hashlib.sha256(
            b"".join(
                v.detach().cpu().numpy().tobytes() for v in model.state_dict().values()
            )
        ).hexdigest()
    )
    fixed = args.mode == "fixed"
    weight = 10.0 if args.mode == "resample10" else 1.0
    validation = problem.sample(4096, 1024, np.random.default_rng(9101))
    rng = np.random.default_rng(2001)
    train = problem.sample(1024, 256, rng, fixed)
    d = problem.d
    n = len(problem.v)
    A = coo_matrix((d["av"], (d["ai"], d["aj"])), shape=(n, n)).tocsc()
    lu = spilu(A)
    # For this small instance, form fixed left-preconditioned arrays; gradients exact.
    pa = torch.as_tensor(lu.solve(A.toarray()), device=DEVICE)
    pb = torch.as_tensor(lu.solve(d["b"]), device=DEVICE)
    nodes = tensor(problem.v)
    femcheck = float(np.linalg.norm(A @ d["truth"] - d["b"]) / np.linalg.norm(d["b"]))
    assert femcheck < 1e-9
    manifest = {
        **vars(args),
        "sample_id": 1,
        "train_row": 449,
        "network": "2-64x6-2 tanh, coordinate-only",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial_hash": initial,
        "instance_sha256": hashlib.sha256(
            (ROOT / "instance.npz").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "k_per_um": K,
        "lambda_b": weight,
        "normalized": not fixed,
        "adam_collocation": 1024,
        "sampling": (
            "fixed original triangle centroids / edge midpoints"
            if fixed
            else "material-balanced, area-uniform random; every 100 Adam steps; fixed L-BFGS points"
        ),
        "selection": "minimum independent normalized physical residual; no FEM labels; ILU final checkpoint",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "slurm_job": os.getenv("SLURM_JOB_ID"),
        "checks": checks,
        "saved_fem_relative_residual": femcheck,
        "scope": "single-instance development experiment on original training set; not an operator-generalization result",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    best = float("inf")
    best_state = None
    best_step = None
    count = 0
    start = time.monotonic()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.steps, eta_min=1e-5
    )

    def objective():
        if args.mode == "ilu":
            z = model(nodes)
            u = torch.complex(z[:, 0], z[:, 1])
            return (pa @ u - pb).abs().square().mean()
        a, b = problem.residual(model, train)
        return a + weight * b if fixed else a / K**4 + weight * b / K**2

    f = (out / "loss.csv").open("w", newline="")
    writer = csv.writer(f)
    writer.writerow(
        [
            "stage",
            "step",
            "objective",
            "validation_domain_normalized",
            "validation_boundary_normalized",
            "validation_sum",
            "seconds",
        ]
    )

    def record(stage, step, loss):
        nonlocal best, best_state, best_step
        a, b = problem.residual(model, validation)
        aa = float(a.detach() / K**4)
        bb = float(b.detach() / K**2)
        score = aa + bb
        if score < best:
            best = score
            best_state = copy.deepcopy(model.state_dict())
            best_step = f"{stage}:{step}"
            torch.save(best_state, out / "best_physics.pth")
        writer.writerow([stage, step, loss, aa, bb, score, time.monotonic() - start])
        f.flush()
        print(
            f"{stage} {step}: objective={loss:.6g} independent={score:.6g}", flush=True
        )

    record("initial", 0, float(objective().detach()))
    for step in range(1, args.steps + 1):
        if not fixed and args.mode != "ilu" and step % 100 == 0:
            train = problem.sample(1024, 256, rng)
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % 250 == 0:
            record("Adam", step, float(loss.detach()))
    torch.save(model.state_dict(), out / "adam_final.pth")
    if not fixed:
        train = problem.sample(4096, 512, rng)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=args.lbfgs,
        history_size=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
    )

    def closure():
        nonlocal count
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite LBFGS loss")
        loss.backward()
        count += 1
        if count % 100 == 0:
            record("LBFGS_evaluation", count, float(loss.detach()))
        return loss

    optimizer.step(closure)
    record("final", count, float(objective().detach()))
    f.close()
    torch.save(model.state_dict(), out / "final.pth")
    results = {}
    for label, state in [("final", model.state_dict()), ("best_physics", best_state)]:
        # Clone final before loading the independently selected checkpoint.
        model.load_state_dict(state)
        with torch.no_grad():
            z = model(nodes)
            p = torch.complex(z[:, 0], z[:, 1]).cpu().numpy()
        truth = (
            np.exp(1j * (problem.v[:, 0] + 0.5 * problem.v[:, 1]))
            if problem.manufactured
            else d["truth"]
        )
        error = p - truth
        results[label] = {
            "mse": float(np.mean(abs(error) ** 2)),
            "relative_l2_nodes": float(np.linalg.norm(error) / np.linalg.norm(truth)),
            "complex_correlation": float(
                abs(np.vdot(truth, p)) / (np.linalg.norm(truth) * np.linalg.norm(p))
            ),
        }
        savemat(
            out / f"prediction_{label}.mat",
            {
                "prediction": p,
                "truth": truth,
                "vertices": problem.v,
                "triangles": problem.t + 1,
            },
        )
    results.update(
        training_seconds=time.monotonic() - start,
        best_physics_score=best,
        best_step=best_step,
        lbfgs_evaluations=count,
    )
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
