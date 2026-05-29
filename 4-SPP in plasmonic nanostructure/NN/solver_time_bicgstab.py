"""
Solve Ax = b on the test set with scipy BiCGSTAB;
stop when MSE mean(|x-Ez|^2) vs Ez_test is below threshold; report per-sample solve time.

SPP14 note: Ez and b are often stored as (Mmax, 2) real/imag; active DOFs are the first Mi (same as cnn_branch_test3).
"""

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from scipy.sparse.linalg import LinearOperator, bicgstab, spilu
from tqdm.auto import tqdm

from getdata import GetDataset

DATA_DIR = Path(__file__).resolve().parent.parent / "samples and post proceeding" / "mat_data"

try:
    import h5py
except ImportError:
    h5py = None


@dataclass
class SolverTimeConfig:
    matpath: str = str(DATA_DIR / "deepOnet_data_SPP14_41")
    warmup_samples: int = 1
    mse_tol: float = 1.7e-4
    maxiter: int | None = None
    use_ilu: bool = True


def _get_required_key(data, candidates):
    for key in candidates:
        if key in data:
            return data[key]
    raise KeyError(f"Missing required field; candidates: {candidates}")


def _resolve_matpath(matpath: str) -> Path:
    path = Path(matpath).expanduser()
    if path.is_file():
        return path
    candidate = Path(str(path) + ".mat")
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Data file not found: {path.resolve()} or {candidate.resolve()} (cwd={os.getcwd()})"
    )


def _matlab_complex_to_numpy(arr):
    if isinstance(arr, np.ndarray) and arr.dtype.fields is not None:
        field_names = set(arr.dtype.fields.keys())
        if "real" in field_names and "imag" in field_names:
            return arr["real"] + 1j * arr["imag"]
        if "r" in field_names and "i" in field_names:
            return arr["r"] + 1j * arr["i"]
    return arr


def _normalize_mat_array(arr):
    arr = _matlab_complex_to_numpy(np.array(arr))
    if arr.ndim > 1:
        arr = np.transpose(arr, axes=tuple(range(arr.ndim - 1, -1, -1)))
    return np.ascontiguousarray(arr)


def _load_mat_v73(matpath):
    if h5py is None:
        raise ImportError("h5py required for MATLAB v7.3: pip install h5py")
    data = {}
    with h5py.File(matpath, "r") as f:
        for key in f.keys():
            if key.startswith("#"):
                continue
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                data[key] = _normalize_mat_array(obj[()])
            elif isinstance(obj, h5py.Group):
                if "real" in obj and "imag" in obj:
                    data[key] = _matlab_complex_to_numpy(
                        obj["real"][()] + 1j * obj["imag"][()]
                    )
    return data


def _load_mat_compat(matpath: str):
    path = _resolve_matpath(matpath)
    if h5py is not None:
        try:
            if h5py.is_hdf5(str(path)):
                return _load_mat_v73(str(path))
        except Exception:
            pass
    try:
        return loadmat(str(path))
    except NotImplementedError:
        if h5py is None:
            raise ImportError("h5py required for MATLAB v7.3: pip install h5py")
        return _load_mat_v73(str(path))
    except (ValueError, OSError):
        return loadmat(str(path), verify_compressed_data_integrity=False)


def load_test_dataset(matpath: str) -> GetDataset:
    data_set = _load_mat_compat(matpath)
    epsilon_train = _get_required_key(data_set, ["Eplison_train", "Epsilon_train"])
    n_train = len(epsilon_train)
    epsilon_test = _get_required_key(data_set, ["Eplison_test", "Epsilon_test"])
    test_set = GetDataset(
        epsilon_test,
        data_set["X_test"],
        data_set["Ez_test"],
        None,
        data_set["Lambda_test"],
        data_set["Ai_test"],
        data_set["Aj_test"],
        data_set["Av_test"],
        data_set["b_test"],
        data_set["coord_len_test"],
        index_offset=n_train,
    )
    return test_set


def _as_complex_array(x) -> np.ndarray:
    """Same as cnn_branch_test3._as_complex_array."""
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return np.asarray(x, dtype=np.complex128)
    if x.ndim >= 1 and x.shape[-1] == 2:
        return (x[..., 0] + 1j * x[..., 1]).astype(np.complex128)
    return x.astype(np.complex128)


def extract_active_dofs(field, mi: int, mmax: int) -> np.ndarray:
    """First Mi active DOFs (padding along Mmax)."""
    field = _as_complex_array(field)
    field = np.squeeze(field)
    if field.ndim > 1 and field.shape[-1] != 2:
        field = field.reshape(-1)
    if field.ndim == 2 and field.shape[-1] == 2:
        field = field[..., 0] + 1j * field[..., 1]
    field = field.ravel()
    if field.size < mmax:
        mmax = field.size
    mask = np.arange(mmax) < mi
    return field[:mmax][mask].astype(np.complex128)


def build_sparse_system(ai, aj, av, b_vec, mi: int, mmax: int):
    ai = np.asarray(ai).ravel()
    aj = np.asarray(aj).ravel()
    av = _as_complex_array(av).ravel()
    b_rhs = extract_active_dofs(b_vec, mi, mmax)

    mask = (ai > 0) & (aj > 0)
    rows = (ai[mask] - 1).astype(np.int64)
    cols = (aj[mask] - 1).astype(np.int64)
    vals = av[mask]
    a = sp.coo_matrix((vals, (rows, cols)), shape=(mi, mi)).tocsc()
    return a, b_rhs


class EzMSEConverged(Exception):
    pass


def reference_ez(ez_tensor, mi: int, mmax: int) -> np.ndarray:
    return extract_active_dofs(ez_tensor.numpy(), mi, mmax)


def mse_vs_ez(x: np.ndarray, ez_ref: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.complex128).ravel()
    ez_ref = np.asarray(ez_ref, dtype=np.complex128).ravel()
    return float(np.mean(np.abs(x - ez_ref) ** 2))


def relative_residual(a: sp.spmatrix, x: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    r = a @ x - b
    return float(np.linalg.norm(r) / (np.linalg.norm(b) + eps))


def solve_sample_bicgstab(
    ai, aj, av, b_vec, mi: int, mmax: int,
    ez_ref: np.ndarray,
    mse_tol: float = 1e-6,
    maxiter: int | None = None,
    use_ilu: bool = True,
):
    a, b_rhs = build_sparse_system(ai, aj, av, b_vec, mi, mmax)
    cap = maxiter if maxiter is not None else max(2000, mi * 100)
    n_iter = 0
    last_x = [None]

    m_op = None
    if use_ilu:
        try:
            ilu = spilu(a.tocsc(), drop_tol=1e-8, fill_factor=20.0)
            m_op = LinearOperator(a.shape, matvec=ilu.solve)
        except Exception:
            m_op = None

    def callback(xk):
        nonlocal n_iter
        n_iter += 1
        last_x[0] = np.asarray(xk, dtype=np.complex128)
        if mse_vs_ez(last_x[0], ez_ref) < mse_tol:
            raise EzMSEConverged()

    x = None
    info = -1
    try:
        x, info = bicgstab(
            a, b_rhs,
            rtol=1e-10,
            atol=1e-12,
            maxiter=cap,
            callback=callback,
            M=m_op,
        )
    except EzMSEConverged:
        x = last_x[0]
        info = 0

    if x is None:
        x = last_x[0] if last_x[0] is not None else np.zeros(mi, dtype=np.complex128)
    x = np.asarray(x, dtype=np.complex128).ravel()
    mse = mse_vs_ez(x, ez_ref)
    rel_res = relative_residual(a, x, b_rhs)
    return x, info, mse, rel_res, n_iter


def stat_test_solver_time(
    test_set: GetDataset,
    warmup_samples: int = 1,
    mse_tol: float = 1e-6,
    maxiter: int | None = None,
    use_ilu: bool = True,
):
    n_test = len(test_set)
    mmax = int(test_set.coord.shape[1])
    per_sample_times = []
    per_sample_mse = []
    per_sample_rel_res = []
    per_sample_info = []

    for idx in tqdm(range(n_test), desc="BiCGSTAB Ax=b (test)"):
        ai = test_set.Ai[idx].numpy()
        aj = test_set.Aj[idx].numpy()
        av = test_set.Av[idx].numpy()
        b_vec = test_set.b[idx].numpy()
        mi = int(test_set.coord_len[idx].item())
        ez_ref = reference_ez(test_set.Ez[idx], mi, mmax)

        t0 = time.perf_counter()
        _, info, mse, rel_res, _ = solve_sample_bicgstab(
            ai, aj, av, b_vec, mi, mmax, ez_ref,
            mse_tol=mse_tol, maxiter=maxiter, use_ilu=use_ilu,
        )
        per_sample_times.append(time.perf_counter() - t0)
        per_sample_mse.append(mse)
        per_sample_rel_res.append(rel_res)
        per_sample_info.append(info)

    if warmup_samples > 0 and n_test > warmup_samples:
        timed_times = per_sample_times[warmup_samples:]
        timed_mse = per_sample_mse[warmup_samples:]
        timed_res = per_sample_rel_res[warmup_samples:]
    else:
        timed_times = per_sample_times
        timed_mse = per_sample_mse
        timed_res = per_sample_rel_res

    total_time = float(np.sum(timed_times))
    n_timed = len(timed_times)
    avg_time = total_time / n_timed if n_timed > 0 else 0.0
    mse_arr = np.asarray(timed_mse)
    res_arr = np.asarray(timed_res)
    n_converged = int(np.sum(mse_arr < mse_tol))

    print(f"Solver: scipy.sparse.linalg.bicgstab (ILU preconditioner: {use_ilu})")
    print(f"MSE stop threshold: {mse_tol:.1e}  (mean(|x-Ez_test|^2), active Mi DOFs)")
    print(f"Mmax (padded columns): {mmax}")
    print(f"Test samples: {n_test}")
    print(f"Warmup samples: {warmup_samples}")
    print(f"Timed samples: {n_timed}")
    print(f"Total solve time: {total_time:.6f} s")
    print(f"Mean solve time per sample: {avg_time:.6f} s ({avg_time * 1000:.3f} ms)")
    print(f"Min/max per-sample time: {min(timed_times):.6f} / {max(timed_times):.6f} s")
    print(f"MSE(x, Ez_test) — mean: {mse_arr.mean():.3e}, max: {mse_arr.max():.3e}")
    print(f"||Ax-b||/||b|| — mean: {res_arr.mean():.3e}, max: {res_arr.max():.3e}")
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
        "per_sample_rel_residual": per_sample_rel_res,
    }


def main():
    parser = argparse.ArgumentParser(description="Solve test set Ax=b with BiCGSTAB and report timing")
    parser.add_argument("--matpath", type=str, default=SolverTimeConfig.matpath)
    parser.add_argument("--warmup", type=int, default=SolverTimeConfig.warmup_samples)
    parser.add_argument("--mse-tol", type=float, default=SolverTimeConfig.mse_tol)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--no-ilu", action="store_true", help="Disable ILU preconditioner")
    args = parser.parse_args()

    test_set = load_test_dataset(args.matpath)
    stat_test_solver_time(
        test_set,
        warmup_samples=args.warmup,
        mse_tol=args.mse_tol,
        maxiter=args.maxiter,
        use_ilu=not args.no_ilu,
    )


if __name__ == "__main__":
    main()
