# -*- coding: utf-8 -*-
"""
Robin VPINN (boundary via loss_bc, not merged into weak-form edge integral)

Scattering-field equation:
    nabla^2 u_s + k0^2 n^2 u_s = -k0^2 (n^2-n_bg^2) E_inc
where E_inc = exp(-i k0 x)

Training loss:
    loss = loss_var + w_bc * loss_bc

- loss_var: interior weak-form residual only (no boundary terms)
- loss_bc : outer-boundary Robin residual
            d u_s/d n - i k0 u_s = 0
            real/imag split:
                Re: d u_r/d n + k0 u_i = 0
                Im: d u_i/d n - k0 u_r = 0
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from torch.autograd import grad


class ScatteringPINN(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=6):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=-1)
        out = self.net(xy)
        return out[:, 0:1], out[:, 1:2]


def incident_field(x, k0):
    return torch.cos(k0 * x), -torch.sin(k0 * x)


def load_mesh(mat_file):
    data = sio.loadmat(mat_file)
    nodes = data["Nodes"].T
    elements = data["Elements"].T - 1
    return nodes, elements


def normalize_nodes_to_um(nodes):
    span = float(max(nodes[:, 0].max() - nodes[:, 0].min(), nodes[:, 1].max() - nodes[:, 1].min()))
    if span < 1e-3:
        return nodes * 1e6, "m->um"
    return nodes, "um"


def tri_gauss_rule_4pt():
    xi = np.array([1.0 / 3.0, 0.6, 0.2, 0.2], dtype=np.float64)
    eta = np.array([1.0 / 3.0, 0.2, 0.6, 0.2], dtype=np.float64)
    w = np.array([-0.28125, 0.260416666666667, 0.260416666666667, 0.260416666666667], dtype=np.float64)
    return xi, eta, w


def edge_gauss_rule_2pt():
    s = np.array([0.2113248654051871, 0.7886751345948129], dtype=np.float64)
    w = np.array([0.5, 0.5], dtype=np.float64)
    return s, w


def extract_boundary_edges(elements):
    edge_counter = {}
    for tri in elements:
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for a, b in edges:
            e = (a, b) if a < b else (b, a)
            edge_counter[e] = edge_counter.get(e, 0) + 1
    b_edges = [e for e, c in edge_counter.items() if c == 1]
    return np.array(b_edges, dtype=np.int64)


def boundary_nodes_from_edges(boundary_edges, n_nodes, device):
    """Boundary node indices and interior-node mask from boundary edges."""
    b_nodes_np = np.unique(boundary_edges.reshape(-1))
    b_nodes = torch.tensor(b_nodes_np, dtype=torch.long, device=device)
    interior_mask = torch.ones(n_nodes, dtype=torch.bool, device=device)
    interior_mask[b_nodes] = False
    return b_nodes, interior_mask


def precompute_volume(nodes, elements, k0, n_bg, n_scatter, r_scatter, xi, eta, wq, device):
    ne = elements.shape[0]
    nn = nodes.shape[0]
    nq = len(wq)

    f_real = np.zeros(nn, dtype=np.float64)
    f_imag = np.zeros(nn, dtype=np.float64)

    pts_all, phi_all, dphix_all, dphiy_all = [], [], [], []
    detj_all, n2_all, nodes_all = [], [], []

    dN_duv = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    n_bg2 = n_bg ** 2
    n_sc2 = n_scatter ** 2
    r2 = r_scatter ** 2

    inside_count = 0
    total_count = 0

    for tri in elements:
        p0 = nodes[tri[0]]
        p1 = nodes[tri[1]]
        p2 = nodes[tri[2]]

        # Standard Jacobian
        J = np.array(
            [[p1[0] - p0[0], p2[0] - p0[0]],
             [p1[1] - p0[1], p2[1] - p0[1]]],
            dtype=np.float64,
        )
        detJ = abs(np.linalg.det(J))
        invJ = np.linalg.inv(J)
        dN_dxy = dN_duv @ invJ

        pts_e = np.zeros((nq, 2), dtype=np.float64)
        phi_e = np.zeros((nq, 3), dtype=np.float64)
        dphix_e = np.zeros((nq, 3), dtype=np.float64)
        dphiy_e = np.zeros((nq, 3), dtype=np.float64)
        n2_e = np.zeros((nq,), dtype=np.float64)

        for q in range(nq):
            u, v, w = xi[q], eta[q], wq[q]
            N1, N2, N3 = 1.0 - u - v, u, v
            phi_e[q, :] = [N1, N2, N3]
            dphix_e[q, :] = dN_dxy[:, 0]
            dphiy_e[q, :] = dN_dxy[:, 1]

            px = p0[0] + u * (p1[0] - p0[0]) + v * (p2[0] - p0[0])
            py = p0[1] + u * (p1[1] - p0[1]) + v * (p2[1] - p0[1])
            pts_e[q, :] = [px, py]

            in_scatter = (px * px + py * py) <= r2
            n2_loc = n_sc2 if in_scatter else n_bg2
            n2_e[q] = n2_loc
            inside_count += int(in_scatter)
            total_count += 1

            inc_re = np.cos(k0 * px)
            inc_im = -np.sin(k0 * px)
            coeff = k0 ** 2 * (n2_loc - n_bg2)
            src_re = coeff * inc_re
            src_im = coeff * inc_im
            for j in range(3):
                f_real[tri[j]] += w * detJ * src_re * phi_e[q, j]
                f_imag[tri[j]] += w * detJ * src_im * phi_e[q, j]

        pts_all.append(pts_e)
        phi_all.append(phi_e)
        dphix_all.append(dphix_e)
        dphiy_all.append(dphiy_e)
        detj_all.append(np.full((nq,), detJ, dtype=np.float64))
        n2_all.append(n2_e)
        nodes_all.append(np.tile(tri[None, :], (nq, 1)))

    pts_all = np.concatenate(pts_all, axis=0)
    phi_all = np.concatenate(phi_all, axis=0)
    dphix_all = np.concatenate(dphix_all, axis=0)
    dphiy_all = np.concatenate(dphiy_all, axis=0)
    detj_all = np.concatenate(detj_all, axis=0)
    n2_all = np.concatenate(n2_all, axis=0)
    nodes_all = np.concatenate(nodes_all, axis=0)

    return (
        torch.tensor(pts_all, dtype=torch.float64, device=device),
        torch.tensor(phi_all, dtype=torch.float64, device=device),
        torch.tensor(dphix_all, dtype=torch.float64, device=device),
        torch.tensor(dphiy_all, dtype=torch.float64, device=device),
        torch.tensor(detj_all, dtype=torch.float64, device=device),
        torch.tensor(n2_all, dtype=torch.float64, device=device),
        torch.tensor(nodes_all, dtype=torch.long, device=device),
        torch.tensor(f_real, dtype=torch.float64, device=device),
        torch.tensor(f_imag, dtype=torch.float64, device=device),
        inside_count,
        total_count,
    )


def precompute_boundary_for_bc(nodes, b_edges, s1d, w1d, device):
    """Precompute boundary points, outward normals, and weights for loss_bc."""
    b_pts = []
    b_w = []
    b_nx = []
    b_ny = []

    for e in b_edges:
        i, j = int(e[0]), int(e[1])
        p0 = nodes[i]
        p1 = nodes[j]
        edge_vec = p1 - p0
        edge_len = float(np.linalg.norm(edge_vec))
        mid = 0.5 * (p0 + p1)

        # Candidate normals; pick outward (relative to origin)
        n1 = np.array([edge_vec[1], -edge_vec[0]], dtype=np.float64) / max(edge_len, 1e-15)
        n2 = -n1
        n = n1 if np.dot(n1, mid) > np.dot(n2, mid) else n2

        for q in range(len(w1d)):
            s, w = s1d[q], w1d[q]
            pq = (1.0 - s) * p0 + s * p1
            b_pts.append(pq)
            b_w.append(w * edge_len)
            b_nx.append(n[0])
            b_ny.append(n[1])

    return (
        torch.tensor(np.array(b_pts), dtype=torch.float64, device=device),
        torch.tensor(np.array(b_w), dtype=torch.float64, device=device),
        torch.tensor(np.array(b_nx), dtype=torch.float64, device=device),
        torch.tensor(np.array(b_ny), dtype=torch.float64, device=device),
    )


def vpinn_losses(
    model,
    vol_pts,
    vol_phi,
    vol_dphix,
    vol_dphiy,
    vol_detj,
    vol_n2,
    vol_nodes,
    f_real,
    f_imag,
    tri_w,
    bc_pts,
    bc_w,
    bc_nx,
    bc_ny,
    interior_mask,
    k0,
):
    # ---------- Interior weak-form loss (no boundary integral) ----------
    x = vol_pts[:, 0:1].clone().detach().requires_grad_(True)
    y = vol_pts[:, 1:2].clone().detach().requires_grad_(True)

    u_re, u_im = model(x, y)

    u_re_x = grad(u_re, x, grad_outputs=torch.ones_like(u_re), create_graph=True)[0]
    u_re_y = grad(u_re, y, grad_outputs=torch.ones_like(u_re), create_graph=True)[0]
    u_im_x = grad(u_im, x, grad_outputs=torch.ones_like(u_im), create_graph=True)[0]
    u_im_y = grad(u_im, y, grad_outputs=torch.ones_like(u_im), create_graph=True)[0]

    nq = len(tri_w)
    ne = vol_pts.shape[0] // nq
    w_flat = torch.tensor(tri_w, dtype=torch.float64, device=vol_pts.device).repeat(ne)
    vol_fac = (w_flat * vol_detj)[:, None]

    grad_re = (u_re_x * vol_dphix + u_re_y * vol_dphiy) * vol_fac
    grad_im = (u_im_x * vol_dphix + u_im_y * vol_dphiy) * vol_fac
    mass_re = -k0 ** 2 * vol_n2[:, None] * u_re * vol_phi * vol_fac
    mass_im = -k0 ** 2 * vol_n2[:, None] * u_im * vol_phi * vol_fac

    contrib_re = grad_re + mass_re
    contrib_im = grad_im + mass_im

    nnode = f_real.shape[0]
    lhs_re = torch.zeros(nnode, dtype=torch.float64, device=vol_pts.device)
    lhs_im = torch.zeros(nnode, dtype=torch.float64, device=vol_pts.device)
    lhs_re.scatter_add_(0, vol_nodes.flatten(), contrib_re.flatten())
    lhs_im.scatter_add_(0, vol_nodes.flatten(), contrib_im.flatten())

    res_re = lhs_re - f_real
    res_im = lhs_im - f_imag

    # Interior-only weak residual; boundary nodes handled entirely by loss_bc
    if torch.any(interior_mask):
        res_re_in = res_re[interior_mask]
        res_im_in = res_im[interior_mask]
        f_re_in = f_real[interior_mask]
        f_im_in = f_imag[interior_mask]
    else:
        # Fallback if no interior nodes (avoid NaN)
        res_re_in, res_im_in = res_re, res_im
        f_re_in, f_im_in = f_real, f_imag

    denom_var = torch.mean(f_re_in ** 2 + f_im_in ** 2) + 1e-14
    loss_var = torch.mean(res_re_in ** 2 + res_im_in ** 2) / denom_var

    # ---------- Robin boundary loss ----------
    xb = bc_pts[:, 0:1].clone().detach().requires_grad_(True)
    yb = bc_pts[:, 1:2].clone().detach().requires_grad_(True)
    ub_re, ub_im = model(xb, yb)

    ub_re_x = grad(ub_re, xb, grad_outputs=torch.ones_like(ub_re), create_graph=True)[0]
    ub_re_y = grad(ub_re, yb, grad_outputs=torch.ones_like(ub_re), create_graph=True)[0]
    ub_im_x = grad(ub_im, xb, grad_outputs=torch.ones_like(ub_im), create_graph=True)[0]
    ub_im_y = grad(ub_im, yb, grad_outputs=torch.ones_like(ub_im), create_graph=True)[0]

    dun_re = ub_re_x * bc_nx[:, None] + ub_re_y * bc_ny[:, None]
    dun_im = ub_im_x * bc_nx[:, None] + ub_im_y * bc_ny[:, None]

    # d u/d n - i k0 u = 0 -> real/imag
    bc_res_re = dun_re - k0 * ub_im
    bc_res_im = dun_im + k0 * ub_re

    # Edge-quadrature weighted mean square
    w_sum = torch.sum(bc_w) + 1e-14
    loss_bc = torch.sum((bc_res_re.squeeze() ** 2 + bc_res_im.squeeze() ** 2) * bc_w) / w_sum

    return loss_var, loss_bc, res_re, res_im


def train_model(
    model,
    vol_pts,
    vol_phi,
    vol_dphix,
    vol_dphiy,
    vol_detj,
    vol_n2,
    vol_nodes,
    f_real,
    f_imag,
    tri_w,
    bc_pts,
    bc_w,
    bc_nx,
    bc_ny,
    interior_mask,
    k0,
    epochs=12000,
    lr=1e-3,
    w_bc=1.0,
):
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=600, min_lr=1e-6
    )

    history = {"epoch": [], "loss": [], "loss_var": [], "loss_bc": []}

    for ep in range(epochs):
        optimizer.zero_grad()

        loss_var, loss_bc, res_re, res_im = vpinn_losses(
            model,
            vol_pts,
            vol_phi,
            vol_dphix,
            vol_dphiy,
            vol_detj,
            vol_n2,
            vol_nodes,
            f_real,
            f_imag,
            tri_w,
            bc_pts,
            bc_w,
            bc_nx,
            bc_ny,
            interior_mask,
            k0,
        )

        loss = loss_var + w_bc * loss_bc
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if ep % 10 == 0 or ep == epochs - 1:
            history["epoch"].append(ep)
            history["loss"].append(loss.item())
            history["loss_var"].append(loss_var.item())
            history["loss_bc"].append(loss_bc.item())
            if ep % 1000 == 0 or ep == epochs - 1:
                rms_re = torch.sqrt(torch.mean(res_re ** 2)).item()
                rms_im = torch.sqrt(torch.mean(res_im ** 2)).item()
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {ep:6d}: Loss={loss.item():.4e}, "
                    f"Var={loss_var.item():.4e}, BC={loss_bc.item():.4e}, "
                    f"RMS(Re)={rms_re:.3e}, RMS(Im)={rms_im:.3e}, lr={lr_now:.2e}"
                )
            

    print("Training complete")
    print(f"Total training time: {time.time() - start_time:.2f} s")
    return history


def visualize_normE(model, nodes, wavelength=1.55, resolution=250):
    device = next(model.parameters()).device
    k0 = 2.0 * np.pi / wavelength

    x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
    y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()

    gx = torch.linspace(x_min, x_max, resolution)
    gy = torch.linspace(y_min, y_max, resolution)
    X, Y = torch.meshgrid(gx, gy, indexing="ij")
    Xf = X.reshape(-1, 1).to(device=device, dtype=torch.float64)
    Yf = Y.reshape(-1, 1).to(device=device, dtype=torch.float64)

    with torch.no_grad():
        us_re, us_im = model(Xf, Yf)
        inc_re, inc_im = incident_field(Xf, k0)
        ut_re = us_re + inc_re
        ut_im = us_im + inc_im
        normE = torch.sqrt(ut_re ** 2 + ut_im ** 2)

    normE = normE.reshape(resolution, resolution).cpu().numpy()

    plt.figure(figsize=(6.6, 5.4))
    im = plt.imshow(normE.T, origin="lower", extent=[x_min, x_max, y_min, y_max], cmap="jet")
    plt.title("normE")
    plt.xlabel("x (um)")
    plt.ylabel("y (um)")
    plt.colorbar(im)
    plt.tight_layout()
    result_path = "SBC_vpinn_normE.png"
    plt.savefig(result_path, dpi=300, bbox_inches="tight")
    print(f"Result plot saved: {result_path}")
    plt.show()
    plt.close()


def visualize_loss_curve(history, save_name="Robin_vpinn_loss_curve.png"):
    if len(history["epoch"]) == 0:
        return
    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["loss"], label="total_loss")
    plt.plot(history["epoch"], history["loss_var"], label="loss_var")
    plt.plot(history["epoch"], history["loss_bc"], label="loss_bc")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SBC VPINN loss curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    print(f"Loss curve saved: {save_name}")
    plt.show()
    plt.close()


def save_E_pred_grid(
    model,
    nodes,
    k0,
    grid_n=128,
    out_dir="SBC vpinn",
    mat_name="E_pred.mat",
):
    """
    Sample total field Ez_total = Ez_sc + Ez_inc on a grid_n x grid_n regular grid over the
    mesh bounding box; save complex E_pred (grid_n, grid_n) to out_dir/mat_name.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    x_min, x_max = float(nodes[:, 0].min()), float(nodes[:, 0].max())
    y_min, y_max = float(nodes[:, 1].min()), float(nodes[:, 1].max())

    x = torch.linspace(x_min, x_max, grid_n, dtype=dtype, device=device)
    y = torch.linspace(y_min, y_max, grid_n, dtype=dtype, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    X_flat = X.reshape(-1, 1)
    Y_flat = Y.reshape(-1, 1)

    model.eval()
    with torch.no_grad():
        Ez_real_sc, Ez_imag_sc = model(X_flat, Y_flat)
        Ez_inc_real, Ez_inc_imag = incident_field(X_flat, k0)
        Ez_total_real = Ez_real_sc + Ez_inc_real
        Ez_total_imag = Ez_imag_sc + Ez_inc_imag
        E_pred = torch.sqrt(Ez_total_real**2 + Ez_total_imag**2)
        E_pred = E_pred.reshape(grid_n, grid_n).detach().cpu().numpy()

    base = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base, out_dir)
    os.makedirs(save_dir, exist_ok=True)
    mat_path = os.path.join(save_dir, mat_name)
    sio.savemat(
        mat_path,
        {
            "E_pred": E_pred,
            "grid_n": np.array(grid_n, dtype=np.int64),
            "x_min": np.array(x_min, dtype=np.float64),
            "x_max": np.array(x_max, dtype=np.float64),
            "y_min": np.array(y_min, dtype=np.float64),
            "y_max": np.array(y_max, dtype=np.float64),
            "k0": np.array(k0, dtype=np.float64),
        },
    )
    print(f"E_pred ({grid_n}x{grid_n}) saved to: {mat_path}")


def main():
    # Physical parameters
    wavelength = 1.55
    k0 = 2.0 * np.pi / wavelength
    n_bg = 1.0
    n_scatter = 1.45
    r_scatter = 0.2

    # Training parameters
    hidden_dim = 128
    num_layers = 6
    epochs = 15000
    lr = 1e-3
    w_bc = 1.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"k0={k0:.8f} 1/um")

    mesh_file = "MeshData_Robin.mat"
    if not os.path.exists(mesh_file):
        if os.path.exists("MeshData_new.mat"):
            mesh_file = "MeshData_new.mat"
        else:
            raise FileNotFoundError("MeshData_Robin.mat or MeshData_new.mat not found")

    nodes, elements = load_mesh(mesh_file)
    nodes, unit_info = normalize_nodes_to_um(nodes)
    print(f"Mesh: nodes={nodes.shape[0]}, elems={elements.shape[0]}, unit={unit_info}")

    # Volume quadrature data
    xi, eta, w_tri = tri_gauss_rule_4pt()
    (
        vol_pts,
        vol_phi,
        vol_dphix,
        vol_dphiy,
        vol_detj,
        vol_n2,
        vol_nodes,
        f_real,
        f_imag,
        inside_count,
        total_count,
    ) = precompute_volume(
        nodes,
        elements,
        k0,
        n_bg,
        n_scatter,
        r_scatter,
        xi,
        eta,
        w_tri,
        device,
    )
    rhs_rms = torch.sqrt(torch.mean(f_real ** 2 + f_imag ** 2)).item()
    print(
        f"Scatterer quadrature ratio: {inside_count}/{total_count} "
        f"({inside_count/max(total_count,1):.3%}), RHS RMS={rhs_rms:.3e}"
    )

    # Boundary quadrature data (for BC loss)
    b_edges = extract_boundary_edges(elements)
    b_node_idx, interior_mask = boundary_nodes_from_edges(b_edges, nodes.shape[0], device)
    s_edge, w_edge = edge_gauss_rule_2pt()
    bc_pts, bc_w, bc_nx, bc_ny = precompute_boundary_for_bc(nodes, b_edges, s_edge, w_edge, device)
    print(
        f"Boundary edges={b_edges.shape[0]}, boundary quad points={bc_pts.shape[0]}, "
        f"boundary nodes={b_node_idx.shape[0]}, interior nodes={int(torch.sum(interior_mask).item())}"
    )

    # Model and training
    model = ScatteringPINN(hidden_dim=hidden_dim, num_layers=num_layers).double().to(device)
    history = train_model(
        model,
        vol_pts,
        vol_phi,
        vol_dphix,
        vol_dphiy,
        vol_detj,
        vol_n2,
        vol_nodes,
        f_real,
        f_imag,
        w_tri,
        bc_pts,
        bc_w,
        bc_nx,
        bc_ny,
        interior_mask,
        k0,
        epochs=epochs,
        lr=lr,
        w_bc=w_bc,
    )

    torch.save(model.state_dict(), "Robin_vpinn_model.pth")
    np.savez(
        "Robin_vpinn_history.npz",
        epoch=np.array(history["epoch"]),
        loss=np.array(history["loss"]),
        loss_var=np.array(history["loss_var"]),
        loss_bc=np.array(history["loss_bc"]),
    )
    visualize_loss_curve(history, save_name="SBC_vpinn_loss_curve.png")

    visualize_normE(model, nodes, wavelength=wavelength, resolution=250)
    save_E_pred_grid(model, nodes, k0=k0, grid_n=128, out_dir="SBC vpinn")


if __name__ == "__main__":
    main()
