"""Audit source."""

from pathlib import Path
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

root = Path(__file__).resolve().parent
d = np.load(root / "instance.npz")
v = d["vertices"]
t = d["triangles"]
p = v[t]
area = abs(np.linalg.det(p[:, 1:] - p[:, :1])) / 2
eps = np.where(d["domain"] == 1, 1.0, 1.45**2)
k = 2 * np.pi / 1.55
A = coo_matrix((d["av"], (d["ai"], d["aj"])), shape=(len(v), len(v))).tocsc()
records = []
bs = []
for order in (8, 12):
    g, w = np.polynomial.legendre.leggauss(order)
    g = (g + 1) / 2
    w = w / 2
    s, q = np.meshgrid(g, g, indexing="ij")
    ww = np.outer(w, w) * 2 * s
    bary = np.stack([1 - s, s * (1 - q), s * q], -1).reshape(-1, 3)
    ww = ww.ravel()
    x = np.einsum("qi,ti->tq", bary, p[:, :, 0])
    f = k * k * (eps[:, None] - 1) * np.exp(-1j * k * x)
    local = area[:, None] * np.einsum("q,tq,qi->ti", ww, f, bary)
    b = np.zeros(len(v), dtype=complex)
    np.add.at(b, t.ravel(), local.ravel())
    bs.append(b)
    u = spsolve(A, b)
    records.append(
        {
            "quadrature_order_per_axis": order,
            "rhs_relative_difference": float(
                np.linalg.norm(b - d["b"]) / np.linalg.norm(d["b"])
            ),
            "solution_mse_from_rhs_difference": float(
                np.mean(abs(u - d["truth"]) ** 2)
            ),
            "solution_relative_l2_from_rhs_difference": float(
                np.linalg.norm(u - d["truth"]) / np.linalg.norm(d["truth"])
            ),
        }
    )
result = {
    "sample_id": 1,
    "records": records,
    "quadrature_convergence_relative": float(
        np.linalg.norm(bs[0] - bs[1]) / np.linalg.norm(bs[1])
    ),
    "interpretation": "Same saved FEM matrix, high-order integration of analytic strong source. Quantifies source mismatch impact for this instance; does not establish its origin.",
}
(root / "source_audit.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
