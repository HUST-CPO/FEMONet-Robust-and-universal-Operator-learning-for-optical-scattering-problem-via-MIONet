"""Exact elementwise L2 integration of linear scalar/edge finite elements.

Uses saved FEM meshes, the repository's edge basis conventions, and periodic
prolongation P before integration. No vertex averaging or magnitude-only error.
"""

from compute_l2 import *


class MatrixReader:
    def __init__(self, path):
        self.h = h5py.File(path) if h5py.is_hdf5(path) else None
        self.m = (
            None
            if self.h
            else loadmat(
                path, variable_names=["Matrix"], struct_as_record=False, squeeze_me=True
            )["Matrix"]
        )

    def get(self, row, sample):
        if self.h:
            return read_h5(self.h, self.h[self.h["Matrix"][sample, row]])
        x = self.m[row, sample]
        if hasattr(x, "_fieldnames"):
            return {k: getattr(x, k) for k in x._fieldnames}
        return x

    def close(self):
        if self.h:
            self.h.close()


def integrate_linear(values, volume):
    """Integral |linear vector field|^2 from its local vertex values.

    values: element x local vertex x vector component x field (error/reference).
    Exact simplex moments: int lambda_i lambda_j = V*(1+delta_ij)/(n*(n+1)).
    """
    n = values.shape[1]
    a = np.sum(np.abs(values) ** 2, axis=(1, 2))
    b = np.sum(np.abs(values.sum(axis=1)) ** 2, axis=1)
    return np.sum(volume[:, None] * (a + b) / (n * (n + 1)), axis=0)


def element_values(reader, sid, case, fields):
    if case in ["1", "2", "3", "OOD"]:
        vertices = reader.get(5, sid)
        tri = reader.get(6, sid).astype(int) - 1
        xy = vertices[tri, :2]
        jac = xy[:, 1:] - xy[:, :1]
        volume = np.abs(np.linalg.det(jac)) / 2
        return fields[tri][:, :, None, :], volume
    mesh = reader.get(6, sid)
    if case == "5":
        fields = reader.get(7, sid) @ fields
        tet = mesh["Tet"].astype(int) - 1
        xyz = mesh["Vertex"][tet]
        jac = xyz[:, :3] - xyz[:, 3:4]
        volume = np.abs(np.linalg.det(jac)) / 6
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        lengths = np.stack(
            [np.linalg.norm(xyz[:, a] - xyz[:, b], axis=1) for a, b in pairs], axis=1
        )
        # Reference vertices (u,v,w)=(1,0,0),(0,1,0),(0,0,1),(0,0,0).
        grads = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]])
        basis = np.empty((4, 3, 6))
        for v in range(4):
            for j, (a, b) in enumerate(pairs):
                basis[v, :, j] = (v == a) * grads[b] - (v == b) * grads[a]
        co = fields[mesh["EdgeOfTet"].astype(int) - 1] * lengths[:, :, None]
    else:
        if case == "SPP1":
            fields = reader.get(7, sid) @ fields
        tri = mesh["tri"].astype(int) - 1
        xy = mesh["vertex"][tri, :2]
        jac = xy[:, 1:] - xy[:, :1]
        volume = np.abs(np.linalg.det(jac)) / 2
        pairs = [(0, 1), (0, 2), (1, 2)]
        lengths = np.stack(
            [np.linalg.norm(xy[:, a] - xy[:, b], axis=1) for a, b in pairs], axis=1
        )
        basis = np.array(
            [[[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 0, 1]], [[0, 1, -1], [0, 1, 0]]],
            dtype=float,
        )
        co = fields[mesh["edgesOfTri"].astype(int) - 1] * lengths[:, :, None]
    local = np.einsum("vce,tef->tvcf", basis, co)
    physical = np.einsum("tij,tvjf->tvif", np.linalg.inv(jac), local)
    assert np.all(volume > 0)
    return physical, volume


def main():
    rows = []
    summary = []
    for cfg in CASES:
        d = ROOT / cfg["folder"]
        split = "test"
        data = variables(
            d / f"deepOnet_data_{cfg['stem']}.mat", ["Ez_test", "coord_len_test"]
        )
        ids = (
            variables(d / f"idx_{cfg['stem']}.mat", ["testIdx"])["testIdx"]
            .ravel()
            .astype(int)
        )
        pp = (
            ROOT / cfg["pred_template"].format(split=split)
            if "pred_template" in cfg
            else d / f"E_test_pred_size_{cfg['pred']}.mat"
        )
        pred = variables(pp, ["E_pred"])["E_pred"]
        reader = MatrixReader(d / f"Train_data_{cfg['stem']}.mat")
        group = []
        for i, (sid, L) in enumerate(
            zip(ids, data["coord_len_test"].ravel().astype(int))
        ):
            t = data["Ez_test"][i, :L] * cfg["scale"]
            p = pred[i, :L] * cfg["scale"]
            stored = reader.get(4, sid - 1).ravel() * cfg["scale"]
            assert np.allclose(stored, t, rtol=1e-11, atol=1e-12), (
                cfg["case"],
                sid,
                "reference mismatch",
            )
            vals, vol = element_values(
                reader, sid - 1, cfg["case"], np.stack([p - t, t], axis=1)
            )
            e, r = integrate_linear(vals, vol)
            row = dict(
                case=cfg["case"],
                method=cfg["method"],
                split=split,
                split_index=i + 1,
                sample_id=int(sid),
                error_integral=float(e),
                reference_integral=float(r),
                domain_measure=float(vol.sum()),
                spatial_relative_l2=float(np.sqrt(e / r)),
                spatial_absolute_l2_native=float(np.sqrt(e)),
                spatial_rmse=float(np.sqrt(e / vol.sum())),
            )
            group.append(row)
            if (i + 1) % 1000 == 0:
                print(cfg["case"], i + 1, "samples integrated", flush=True)
        reader.close()
        rel = np.array([r["spatial_relative_l2"] for r in group])
        err = sum(r["error_integral"] for r in group)
        ref = sum(r["reference_integral"] for r in group)
        s = dict(
            case=cfg["case"],
            method=cfg["method"],
            split=split,
            n_samples=len(group),
            spatial_relative_l2_mean=float(rel.mean()),
            spatial_relative_l2_pooled=float(np.sqrt(err / ref)),
            spatial_relative_l2_std=float(rel.std()),
            spatial_relative_l2_median=float(np.median(rel)),
            spatial_relative_l2_max=float(rel.max()),
            spatial_absolute_l2_native_mean=float(
                np.mean([r["spatial_absolute_l2_native"] for r in group])
            ),
            spatial_rmse_pooled=float(
                np.sqrt(err / sum(r["domain_measure"] for r in group))
            ),
        )
        rows += group
        summary.append(s)
        print(json.dumps(s), flush=True)
        write_csv(OUT / "spatial_per_sample.csv", rows)
        write_csv(OUT / "spatial_summary.csv", summary)
    (OUT / "spatial_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
