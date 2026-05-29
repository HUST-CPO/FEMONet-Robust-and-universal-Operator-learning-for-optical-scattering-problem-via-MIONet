"""
BiCGSTAB solve Ax=b on test set; stop when MSE(x, Ez_test) < tol.
Report per-sample average solve time.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from scipy.sparse.linalg import bicgstab
from tqdm.auto import tqdm

from getdata import GetDataset

DATA_DIR = Path(__file__).resolve().parent.parent / "samples and post proceeding" / "mat_data"


@dataclass
class SolverTimeConfig:
    matpath: str = str(DATA_DIR / "deepOnet_data_A_1558")
    warmup_samples: int = 1
    mse_tol: float = 4.39e-5
    maxiter: int | None = None


def _resolve_matpath(matpath: str) -> str:
    path = Path(matpath)
    if path.suffix != ".mat":
        path = path.with_suffix(".mat")
    return str(path)


def _load_mat_compat(matpath: str):
    matpath = _resolve_matpath(matpath)
    try:
        return loadmat(matpath)
    except NotImplementedError:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py required for MATLAB v7.3: pip install h5py") from exc

        def _to_numpy(arr):
            arr = np.array(arr)
            if arr.dtype.fields and "real" in arr.dtype.fields:
                return arr["real"] + 1j * arr["imag"]
            if arr.ndim > 1:
                arr = np.transpose(arr, tuple(range(arr.ndim - 1, -1, -1)))
            return arr

        data = {}
        with h5py.File(matpath, "r") as f:
            for key in f.keys():
                if not key.startswith("#"):
                    data[key] = _to_numpy(f[key][()])
        return data


def load_test_dataset(matpath: str) -> GetDataset:
    data_set = _load_mat_compat(matpath)
    n_train = len(data_set["Eplison_train"])
    test_set = GetDataset(
        data_set["Eplison_test"],
        data_set["X_test"],
        data_set["Ez_test"],
        data_set["Ai_test"],
        data_set["Aj_test"],
        data_set["Av_test"],
        data_set["b_test"],
        data_set["coord_len_test"],
        index_offset=n_train,
    )
    return test_set


def build_sparse_system(ai, aj, av, b_vec, mi: int):
    ai = np.asarray(ai).ravel()
    aj = np.asarray(aj).ravel()
    av = np.asarray(av).ravel()
    b_vec = np.asarray(b_vec).ravel()[:mi]

    mask = (ai > 0) & (aj > 0)
    rows = (ai[mask] - 1).astype(np.int64)
    cols = (aj[mask] - 1).astype(np.int64)
    vals = av[mask]
    a = sp.coo_matrix((vals, (rows, cols)), shape=(mi, mi)).tocsc()
    return a, np.asarray(b_vec, dtype=np.complex128)


class EzMSEConverged(Exception):
    """MSE vs Ez below threshold; stop BiCGSTAB early."""


def reference_ez(ez_tensor, mi: int) -> np.ndarray:
    """Reference solution on active DOFs from Ez_test (complex)."""
    ez = np.asarray(ez_tensor.numpy()[:mi])
    if np.iscomplexobj(ez):
        return ez.astype(np.complex128).ravel()
    ez = np.reshape(ez, (mi, -1))
    if ez.shape[-1] == 2:
        return (ez[:, 0] + 1j * ez[:, 1]).astype(np.complex128)
    return ez.astype(np.complex128).ravel()


def mse_vs_ez(x: np.ndarray, ez_ref: np.ndarray) -> float:
    """MSE(x, Ez) = mean(|x - Ez|^2)。"""
    x = np.asarray(x, dtype=np.complex128).ravel()
    ez_ref = np.asarray(ez_ref, dtype=np.complex128).ravel()
    return float(np.mean(np.abs(x - ez_ref) ** 2))


def solve_sample_bicgstab(
    ai, aj, av, b_vec, mi: int,
    ez_ref: np.ndarray,
    mse_tol: float = 1e-6,
    maxiter: int | None = None,
):
    a, b_rhs = build_sparse_system(ai, aj, av, b_vec, mi)
    cap = maxiter if maxiter is not None else max(500, mi * 50)
    n_iter = 0
    last_x = [None]

    def callback(xk):
        nonlocal n_iter
        n_iter += 1
        last_x[0] = xk
        if mse_vs_ez(xk, ez_ref) < mse_tol:
            raise EzMSEConverged()

    x = None
    info = -1
    try:
        x, info = bicgstab(
            a, b_rhs,
            rtol=1e-12,
            atol=1e-12,
            maxiter=cap,
            callback=callback,
        )
    except EzMSEConverged:
        x = last_x[0]
        info = 0

    if x is None:
        x = last_x[0] if last_x[0] is not None else np.zeros(mi, dtype=np.complex128)
    mse = mse_vs_ez(x, ez_ref)
    return x, info, mse, n_iter


def stat_test_solver_time(
    test_set: GetDataset,
    warmup_samples: int = 1,
    mse_tol: float = 1e-6,
    maxiter: int | None = None,
):
    n_test = len(test_set)
    per_sample_times = []
    per_sample_mse = []
    per_sample_iters = []

    for idx in tqdm(range(n_test), desc="BiCGSTAB Ax=b (test)"):
        ai = test_set.Ai[idx].numpy()
        aj = test_set.Aj[idx].numpy()
        av = test_set.Av[idx].numpy()
        b_vec = test_set.b[idx].numpy()
        mi = int(test_set.coord_len[idx].item())
        ez_ref = reference_ez(test_set.Ez[idx], mi)

        t0 = time.perf_counter()
        _, info, mse, n_iter = solve_sample_bicgstab(
            ai, aj, av, b_vec, mi, ez_ref, mse_tol=mse_tol, maxiter=maxiter,
        )
        elapsed = time.perf_counter() - t0

        per_sample_times.append(elapsed)
        per_sample_mse.append(mse)
        per_sample_iters.append(n_iter)

    if warmup_samples > 0 and n_test > warmup_samples:
        timed_times = per_sample_times[warmup_samples:]
        timed_mse = per_sample_mse[warmup_samples:]
    else:
        timed_times = per_sample_times
        timed_mse = per_sample_mse

    total_time = float(np.sum(timed_times))
    n_timed = len(timed_times)
    avg_time = total_time / n_timed if n_timed > 0 else 0.0
    mse_arr = np.asarray(timed_mse)
    n_converged = int(np.sum(mse_arr < mse_tol))

    print("Solver: scipy.sparse.linalg.bicgstab")
    print(f"MSE stop threshold: {mse_tol:.1e}  (mean(|x-Ez_test|^2))")
    print(f"Test samples: {n_test}")
    print(f"Warmup samples: {warmup_samples}")
    print(f"Timed samples: {n_timed}")
    print(f"Total solve time: {total_time:.6f} s")
    print(f"Mean solve time per sample: {avg_time:.6f} s ({avg_time * 1000:.3f} ms)")
    print(f"Min/max per-sample time: {min(timed_times):.6f} / {max(timed_times):.6f} s")
    print(f"MSE(x, Ez_test) — mean: {mse_arr.mean():.3e}, max: {mse_arr.max():.3e}")
    print(f"Samples with MSE < {mse_tol:.1e}: {n_converged}/{n_timed}")

    return {
        "total_time_sec": total_time,
        "n_samples": n_test,
        "n_timed_samples": n_timed,
        "avg_time_per_sample_sec": avg_time,
        "mse_tol": mse_tol,
        "n_converged_mse": n_converged,
        "per_sample_times_sec": per_sample_times,
        "per_sample_mse": per_sample_mse,
        "per_sample_iters": per_sample_iters,
    }


def main():
    parser = argparse.ArgumentParser(description="Solve test set Ax=b with BiCGSTAB and report timing")
    parser.add_argument("--matpath", type=str, default=SolverTimeConfig.matpath)
    parser.add_argument("--warmup", type=int, default=SolverTimeConfig.warmup_samples)
    parser.add_argument("--mse-tol", type=float, default=SolverTimeConfig.mse_tol)
    parser.add_argument("--maxiter", type=int, default=None, help="Max iterations (default max(500, 50*Mi))")
    args = parser.parse_args()

    test_set = load_test_dataset(args.matpath)
    stat_test_solver_time(
        test_set,
        warmup_samples=args.warmup,
        mse_tol=args.mse_tol,
        maxiter=args.maxiter,
    )


if __name__ == "__main__":
    main()
