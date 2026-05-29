"""Shared data loading and sparse system assembly (3D case3: epsilon + Lambda, no Ebz)."""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

from getdata import GetDataset

try:
    import h5py
except ImportError:
    h5py = None

DATA_DIR = Path(__file__).resolve().parent.parent / "samples and post proceeding" / "mat_data"


@dataclass
class SolverTimeConfig:
    matpath: str = str(DATA_DIR / "deepOnet_data_3Dcase3_261")
    warmup_samples: int = 1
    mse_tol: float = 1.33e-5
    maxiter: int | None = None


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
