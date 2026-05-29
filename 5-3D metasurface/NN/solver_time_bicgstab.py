"""
Solve Ax = b with BiCGSTAB on the test set.
Stop when MSE between x and Ez_test falls below threshold; report per-sample solve time.
"""

import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab, spilu
from tqdm.auto import tqdm

from getdata import GetDataset
from solver_time import DATA_DIR, SolverTimeConfig, build_sparse_system, load_test_dataset


class MSEConverged(Exception):
    """MSE of current iterate vs. Ez_test is below threshold."""


@dataclass
class BiCGSTABConfig(SolverTimeConfig):
    mse_tol: float = 1.33e-5
    maxiter: int = 10000
    rtol: float = 1e-12
    atol: float = 1e-12
    use_ilu: bool = True
    # Match spilu defaults in cnn_branch_test1_DDP.PINNConfig
    ilu_drop_tol: Optional[float] = 1e-8
    ilu_fill_factor: Optional[float] = 20.0
    ilu_drop_rule: Optional[str] = "basic"
    ilu_permc_spec: Optional[str] = "COLAMD"
    ilu_diag_pivot_thresh: Optional[float] = None
    ilu_relax: Optional[float] = None
    ilu_panel_size: Optional[int] = None
    ilu_options: Optional[Dict[str, Any]] = None


def build_ilu_kwargs(cfg: BiCGSTABConfig) -> Dict[str, Any]:
    """Build spilu kwargs; omit None entries (scipy defaults)."""
    raw = dict(
        drop_tol=cfg.ilu_drop_tol,
        fill_factor=cfg.ilu_fill_factor,
        drop_rule=cfg.ilu_drop_rule,
        permc_spec=cfg.ilu_permc_spec,
        diag_pivot_thresh=cfg.ilu_diag_pivot_thresh,
        relax=cfg.ilu_relax,
        panel_size=cfg.ilu_panel_size,
        options=cfg.ilu_options,
    )
    return {k: v for k, v in raw.items() if v is not None}


def build_ilu_preconditioner(a, ilu_kwargs: Dict[str, Any]) -> LinearOperator:
    """Build BiCGSTAB preconditioner M ≈ A^{-1} from ILU factorization."""
    ilu = spilu(a, **ilu_kwargs)
    return LinearOperator(a.shape, matvec=ilu.solve)


def _ez_real_imag(ez_true: np.ndarray, mi: int):
    """Match cnn_branch_test1_DDP.get_data_loss: last dim of Ez is real/imag."""
    ez = np.asarray(ez_true)[:mi]
    if ez.ndim == 1:
        return np.real(ez), np.imag(ez)
    ez_re = np.real(ez[:, 0])
    ez_im = np.real(ez[:, 1])
    return ez_re, ez_im


def compute_solution_mse(x: np.ndarray, ez_true: np.ndarray, mi: int) -> float:
    """MSE of x vs. first mi points of Ez_test (real MSE + imag MSE)."""
    x = np.asarray(x).ravel()[:mi]
    ez_re, ez_im = _ez_real_imag(ez_true, mi)
    mse_re = np.mean((x.real - ez_re) ** 2)
    mse_im = np.mean((x.imag - ez_im) ** 2)
    return float(mse_re + mse_im)


class MSEStopCallback:
    """BiCGSTAB per-step callback: compute MSE vs. Ez_test; early-stop at threshold."""

    def __init__(self, ez_true: np.ndarray, mi: int, mse_tol: float):
        self.ez_true = ez_true
        self.mi = mi
        self.mse_tol = mse_tol
        self.mse_history: list[float] = []
        self.last_x: np.ndarray | None = None
        self.stopped_by_mse = False

    def __call__(self, xk):
        self.last_x = np.asarray(xk).copy()
        mse = compute_solution_mse(self.last_x, self.ez_true, self.mi)
        self.mse_history.append(mse)
        if mse <= self.mse_tol:
            self.stopped_by_mse = True
            raise MSEConverged()


def solve_sample_bicgstab(
    ai,
    aj,
    av,
    b_vec,
    mi: int,
    ez_true: np.ndarray,
    mse_tol: float,
    maxiter: int,
    rtol: float,
    atol: float,
    use_ilu: bool = True,
    ilu_kwargs: Optional[Dict[str, Any]] = None,
):
    a, b_rhs = build_sparse_system(ai, aj, av, b_vec, mi)
    x0 = np.zeros(mi, dtype=np.complex128)
    callback = MSEStopCallback(ez_true, mi, mse_tol)

    m_op = None
    if use_ilu:
        m_op = build_ilu_preconditioner(a, ilu_kwargs or {})

    info = 0
    try:
        x, info = bicgstab(
            a,
            b_rhs,
            x0=x0,
            M=m_op,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            callback=callback,
        )
    except MSEConverged:
        x = callback.last_x
    except Exception:
        if callback.last_x is not None:
            x = callback.last_x
            info = -1
        else:
            raise

    final_mse = compute_solution_mse(x, ez_true, mi)
    return {
        "x": x,
        "info": int(info),
        "final_mse": final_mse,
        "n_iter": len(callback.mse_history),
        "mse_history": callback.mse_history,
        "stopped_by_mse": callback.stopped_by_mse,
    }


def stat_test_bicgstab_time(
    test_set: GetDataset,
    mse_tol: float = BiCGSTABConfig.mse_tol,
    maxiter: int = BiCGSTABConfig.maxiter,
    rtol: float = BiCGSTABConfig.rtol,
    atol: float = BiCGSTABConfig.atol,
    warmup_samples: int = 1,
    use_ilu: bool = BiCGSTABConfig.use_ilu,
    ilu_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Solve Ax = b per test sample with BiCGSTAB; stop on MSE vs. Ez_test; record timing.

    Returns:
        dict: total/avg/per-sample time, MSE, iteration counts, etc.
    """
    n_test = len(test_set)
    per_sample_times = []
    per_sample_mse = []
    per_sample_iters = []
    stopped_by_mse_count = 0

    for idx in tqdm(range(n_test), desc="BiCGSTAB Ax=b (test)"):
        ai = test_set.Ai[idx].numpy()
        aj = test_set.Aj[idx].numpy()
        av = test_set.Av[idx].numpy()
        b_vec = test_set.b[idx].numpy()/10
        ez_true = test_set.Ez[idx].numpy()/10
        mi = int(test_set.coord_len[idx].item())

        t0 = time.perf_counter()
        result = solve_sample_bicgstab(
            ai,
            aj,
            av,
            b_vec,
            mi,
            ez_true,
            mse_tol,
            maxiter,
            rtol,
            atol,
            use_ilu=use_ilu,
            ilu_kwargs=ilu_kwargs,
        )
        elapsed = time.perf_counter() - t0

        per_sample_times.append(elapsed)
        per_sample_mse.append(result["final_mse"])
        per_sample_iters.append(result["n_iter"])
        if result["stopped_by_mse"]:
            stopped_by_mse_count += 1

    if warmup_samples > 0 and n_test > warmup_samples:
        timed_times = per_sample_times[warmup_samples:]
        timed_mse = per_sample_mse[warmup_samples:]
        timed_iters = per_sample_iters[warmup_samples:]
    else:
        timed_times = per_sample_times
        timed_mse = per_sample_mse
        timed_iters = per_sample_iters

    total_time = float(np.sum(timed_times))
    n_timed = len(timed_times)
    avg_time = total_time / n_timed if n_timed > 0 else 0.0

    print(f"Test samples: {n_test}")
    print(f"ILU preconditioner: {'on' if use_ilu else 'off'}")
    if use_ilu and ilu_kwargs:
        print(f"ILU kwargs: {ilu_kwargs}")
    print(f"MSE stop threshold: {mse_tol}")
    print(f"Max iterations: {maxiter}")
    print(f"Warmup samples: {warmup_samples}")
    print(f"Timed samples: {n_timed}")
    print(f"Samples stopped early by MSE: {stopped_by_mse_count} / {n_test}")
    print(f"Total solve time: {total_time:.6f} s")
    print(f"Mean solve time per sample: {avg_time:.6f} s ({avg_time * 1000:.3f} ms)")
    print(f"Min/max per-sample time: {min(timed_times):.6f} / {max(timed_times):.6f} s")
    print(f"Final MSE — mean: {np.mean(timed_mse):.6e}, max: {np.max(timed_mse):.6e}")
    print(f"Iterations — mean: {np.mean(timed_iters):.1f}, max: {int(np.max(timed_iters))}")

    return {
        "total_time_sec": total_time,
        "n_samples": n_test,
        "n_timed_samples": n_timed,
        "avg_time_per_sample_sec": avg_time,
        "per_sample_times_sec": per_sample_times,
        "per_sample_mse": per_sample_mse,
        "per_sample_iters": per_sample_iters,
        "stopped_by_mse_count": stopped_by_mse_count,
        "mse_tol": mse_tol,
        "use_ilu": use_ilu,
    }


def main():
    parser = argparse.ArgumentParser(
        description="BiCGSTAB solve Ax=b on test set (MSE early stop vs Ez_test) with timing"
    )
    parser.add_argument(
        "--matpath",
        type=str,
        default=BiCGSTABConfig.matpath,
        help="MATLAB data path (.mat suffix optional)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=BiCGSTABConfig.warmup_samples,
        help="Warmup samples excluded from mean time",
    )
    parser.add_argument(
        "--mse-tol",
        type=float,
        default=BiCGSTABConfig.mse_tol,
        help="Stop when MSE(x, Ez_test) falls below this value",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=BiCGSTABConfig.maxiter,
        help="BiCGSTAB max iterations",
    )
    parser.add_argument("--rtol", type=float, default=BiCGSTABConfig.rtol)
    parser.add_argument("--atol", type=float, default=BiCGSTABConfig.atol)
    parser.add_argument(
        "--no-ilu",
        action="store_true",
        help="Disable ILU preconditioner (enabled by default)",
    )
    parser.add_argument(
        "--ilu-drop-tol",
        type=float,
        default=BiCGSTABConfig.ilu_drop_tol,
        help="spilu drop_tol",
    )
    parser.add_argument(
        "--ilu-fill-factor",
        type=float,
        default=BiCGSTABConfig.ilu_fill_factor,
        help="spilu fill_factor",
    )
    args = parser.parse_args()

    cfg = BiCGSTABConfig(
        ilu_drop_tol=args.ilu_drop_tol,
        ilu_fill_factor=args.ilu_fill_factor,
    )
    use_ilu = not args.no_ilu
    ilu_kwargs = build_ilu_kwargs(cfg) if use_ilu else None

    test_set = load_test_dataset(args.matpath)
    stat_test_bicgstab_time(
        test_set,
        mse_tol=args.mse_tol,
        maxiter=args.maxiter,
        rtol=args.rtol,
        atol=args.atol,
        warmup_samples=args.warmup,
        use_ilu=use_ilu,
        ilu_kwargs=ilu_kwargs,
    )


if __name__ == "__main__":
    main()
