"""Prepare collocation."""

import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.sparse import coo_matrix
import hashlib


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


ROOT = Path(__file__).resolve().parent
K = 2 * np.pi / 1.55  # coordinates in micrometres


def main():
    source = ROOT.parent / "A-proceeding"
    path = source / "Train_data_A1_all.mat"
    names = ["Matrix", "Ebz_all", "eps_all"]
    names += [
        f"{a}_{s}" for s in ("train", "test") for a in ("Nodes", "Elements", "Domains")
    ]
    d = loadmat(path, variable_names=names)
    ids = loadmat(source / "idx_A1_all.mat")
    ref = loadmat(source / "deepOnet_data_A1_all.mat")
    out, errors, rhs_errors = {}, [], []
    # Seven-point Dunavant quadrature used only to audit the physical source.
    bary = np.array(
        [
            [1 / 3] * 3,
            [0.05971587178977, 0.470142064105115, 0.470142064105115],
            [0.470142064105115, 0.05971587178977, 0.470142064105115],
            [0.470142064105115, 0.470142064105115, 0.05971587178977],
            [0.797426985353087, 0.101286507323456, 0.101286507323456],
            [0.101286507323456, 0.797426985353087, 0.101286507323456],
            [0.101286507323456, 0.101286507323456, 0.797426985353087],
        ]
    )
    weights = np.array([0.225] + [0.132394152788506] * 3 + [0.125939180544827] * 3)
    xgrid = np.linspace(-1, 1, 128)
    expected = np.exp(-1j * K * xgrid)[None, :]
    inc = d["Ebz_all"][..., 0] + 1j * d["Ebz_all"][..., 1]
    np.testing.assert_allclose(inc, np.broadcast_to(expected, inc.shape), atol=1e-12)
    assert np.max(np.abs(d["eps_all"][..., 1])) == 0
    for split in ("train", "test"):
        records = []
        for j, sid in enumerate(ids[split + "Idx"].ravel().astype(int) - 1):
            v = d["Matrix"][5, sid]
            t = d["Matrix"][6, sid].astype(int) - 1
            np.testing.assert_array_equal(v, d["Nodes_" + split][0, j])
            np.testing.assert_array_equal(t + 1, d["Elements_" + split][0, j])
            n = int(ref["coord_len_" + split][j, 0])
            np.testing.assert_array_equal(v, ref["X_" + split][j, :n])
            np.testing.assert_array_equal(
                d["Matrix"][4, sid].ravel(), ref["Ez_" + split][j, :n]
            )
            np.testing.assert_array_equal(
                d["eps_all"][sid, :, :, 0], ref["Eplison_" + split][j, 0]
            )
            domain = d["Domains_" + split][0, j].ravel()
            assert set(domain) == {1, 2}
            eps = np.where(domain == 1, 1.0, 1.45**2)
            p = v[t]
            jac = p[:, 1:] - p[:, :1]
            area = abs(np.linalg.det(jac)) / 2
            assert np.all(area > 0) and abs(area.sum() - 4) < 1e-10
            grads = np.einsum(
                "tij,jk->tik", np.linalg.inv(jac), np.array([[-1, 1, 0], [-1, 0, 1]])
            )
            local = area[:, None, None] * (
                np.einsum("tki,tkj->tij", grads, grads)
                - K * K * eps[:, None, None] * (np.ones((3, 3)) + np.eye(3)) / 12
            )
            rows = np.broadcast_to(t[:, :, None], local.shape).ravel()
            cols = np.broadcast_to(t[:, None, :], local.shape).ravel()
            edges = np.sort(
                np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1
            )
            edges, counts = np.unique(edges, axis=0, return_counts=True)
            edges = edges[counts == 1]
            bp = v[edges].mean(axis=1)
            lengths = np.linalg.norm(v[edges[:, 1]] - v[edges[:, 0]], axis=1)
            normal = np.zeros_like(bp)
            for axis in (0, 1):
                normal[:, axis] = np.where(
                    np.isclose(abs(bp[:, axis]), 1, atol=1e-12), np.sign(bp[:, axis]), 0
                )
            np.testing.assert_array_equal(np.sum(normal**2, axis=1), np.ones(len(bp)))
            bm = 1j * K * lengths[:, None, None] * (np.ones((2, 2)) + np.eye(2)) / 6
            A = coo_matrix(
                (
                    np.concatenate([local.ravel(), bm.ravel()]),
                    (
                        np.concatenate(
                            [rows, np.broadcast_to(edges[:, :, None], bm.shape).ravel()]
                        ),
                        np.concatenate(
                            [cols, np.broadcast_to(edges[:, None, :], bm.shape).ravel()]
                        ),
                    ),
                ),
                shape=(n, n),
            ).tocsr()
            ai, aj, av = [d["Matrix"][k, sid].ravel() for k in range(3)]
            saved = coo_matrix(
                (av, (ai.astype(int) - 1, aj.astype(int) - 1)), shape=(n, n)
            ).tocsr()
            errors.append(
                float(np.linalg.norm((A - saved).data) / np.linalg.norm(saved.data))
            )
            src = (
                K
                * K
                * (eps[:, None] - 1)
                * np.exp(-1j * K * np.einsum("qi,ti->tq", bary, p[:, :, 0]))
            )
            local_b = area[:, None] * np.einsum("q,tq,qi->ti", weights, src, bary)
            b = np.zeros(n, dtype=complex)
            np.add.at(b, t.ravel(), local_b.ravel())
            saved_b = d["Matrix"][3, sid].ravel()
            rhs_errors.append(
                float(np.linalg.norm(b - saved_b) / np.linalg.norm(saved_b))
            )
            records.append((p.mean(axis=1), eps, bp, normal))
        for col, key in enumerate(("xy", "eps", "boundary", "normal")):
            width = max(len(r[col]) for r in records)
            shape = (len(records), width) + records[0][col].shape[1:]
            a = np.zeros(shape)
            for j, r in enumerate(records):
                a[j, : len(r[col])] = r[col]
            out[split + "_" + key] = a
        out[split + "_interior_len"] = np.array([len(r[0]) for r in records])
        out[split + "_boundary_len"] = np.array([len(r[2]) for r in records])
        out[split + "_sample_ids"] = ids[split + "Idx"].ravel()
    audit = {
        "data_sha256": sha256(source / "deepOnet_data_A1_all.mat"),
        "mesh_sha256": sha256(path),
        "matrix_max_relative_error": max(errors),
        "source_max_relative_error": max(rhs_errors),
        "k_per_um": K,
        "incident": "exp(-i*k*x), amplitude 1; all stored grids verified",
        "material": "domain 1: eps=1; domain 2: eps=1.45^2; lossless",
        "points": "one centroid per original triangle; one midpoint per outer edge",
    }
    print(json.dumps(audit, indent=2), flush=True)
    assert max(errors) < 1e-10, "Material/geometry/boundary operator mismatch"
    # The saved FEM RHS and analytic strong source are not bitwise equivalent:
    # retain and report their discrepancy instead of fitting the source to b.
    audit["source_audit_note"] = (
        "Analytic strong-source quadrature differs from saved FEM b by up to 0.227%; origin not established. No b is used in PINN training."
    )
    assert max(rhs_errors) < 0.003, "Incident/source mismatch exceeds audited tolerance"
    np.savez_compressed(ROOT / "collocation.npz", **out)
    audit["collocation_sha256"] = sha256(ROOT / "collocation.npz")
    (ROOT / "collocation_audit.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
