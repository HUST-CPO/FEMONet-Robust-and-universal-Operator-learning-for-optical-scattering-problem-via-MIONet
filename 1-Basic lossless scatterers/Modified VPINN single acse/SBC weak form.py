# -*- coding: utf-8 -*-
"""
Robin weak-form VPINN for the scattering field.

Weak form (boundary in the weak form; no separate loss_bc):

    int_Omega grad(u_s)·grad(v) dOmega - int_Omega k0^2 n^2 u_s v dOmega
        - int_Gamma i k0 u_s v dGamma = int_Omega f v dOmega

- u_s: scattered field (complex)
- v: test function (real linear basis)
- Gamma: outer boundary (square)
- f = k0^2 (n^2 - n_bg^2) E_inc,  E_inc = exp(-i*k0*x)

Homogeneous Robin (g=0): no boundary RHS. Only weak-form residual loss.
"""

# --- Environment ---
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Dependencies ---
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from torch.autograd import grad


# --- Network ---
class ScatteringPINN(nn.Module):
    """2D (x,y) input; scattered-field real and imaginary parts."""

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


# --- Physics ---
def incident_field(x, k0):
    """E_inc = exp(-i*k0*x) = cos(k0*x) - i sin(k0*x)."""
    inc_real = torch.cos(k0 * x)
    inc_imag = -torch.sin(k0 * x)
    return inc_real, inc_imag


# --- Mesh I/O ---
def load_mesh(mat_file):
    """Load MATLAB mesh: Nodes(2,N), Elements(3,Ne) -> row-major."""
    data = sio.loadmat(mat_file)
    nodes = data["Nodes"].T
    elements = data["Elements"].T - 1
    return nodes, elements


def normalize_nodes_to_um(nodes):
    """Convert mesh length units to um if needed (wavelength=1.55 um)."""
    span_x = float(nodes[:, 0].max() - nodes[:, 0].min())
    span_y = float(nodes[:, 1].max() - nodes[:, 1].min())
    span = max(span_x, span_y)
    if span < 1e-3:
        return nodes * 1e6, "m->um"
    return nodes, "um"


# --- Quadrature rules ---
def tri_gauss_rule_4pt():
    """4-point triangle quadrature (matches test.py)."""
    xi = np.array([1.0 / 3.0, 0.6, 0.2, 0.2], dtype=np.float64)
    eta = np.array([1.0 / 3.0, 0.2, 0.6, 0.2], dtype=np.float64)
    w = np.array([-0.28125, 0.260416666666667, 0.260416666666667, 0.260416666666667], dtype=np.float64)
    return xi, eta, w


def edge_gauss_rule_2pt():
    """2-point Gauss rule on [0,1]."""
    s = np.array([0.2113248654051871, 0.7886751345948129], dtype=np.float64)
    w = np.array([0.5, 0.5], dtype=np.float64)
    return s, w


# --- Boundary edges ---
def extract_boundary_edges(elements):
    """Extract outer boundary edges (appear exactly once)."""
    edge_count = {}
    for tri in elements:
        candidate = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for a, b in candidate:
            e = (a, b) if a < b else (b, a)
            edge_count[e] = edge_count.get(e, 0) + 1
    boundary = [e for e, c in edge_count.items() if c == 1]
    return np.array(boundary, dtype=np.int64)


# --- Volume precompute ---
def precompute_volume(
    nodes,
    elements,
    k0,
    n_bg,
    n_scatter,
    r_scatter,
    xi,
    eta,
    wq,
    device,
):
    """
    Precompute volume quadrature: points, shape functions, detJ, n^2, node map, RHS F.

    Jacobian J = [[dx/du, dx/dv],[dy/du, dy/dv]]; dN/dx = dN/d(u,v) @ inv(J); area factor |detJ|.
    """
    ne = elements.shape[0]
    nn = nodes.shape[0]
    nq = len(wq)

    f_real = np.zeros(nn, dtype=np.float64)
    f_imag = np.zeros(nn, dtype=np.float64)

    all_pts = []
    all_phi = []
    all_dphix = []
    all_dphiy = []
    all_detj = []
    all_n2 = []
    all_nodes = []

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

        J = np.array(
            [
                [p1[0] - p0[0], p2[0] - p0[0]],
                [p1[1] - p0[1], p2[1] - p0[1]],
            ],
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
            u = xi[q]
            v = eta[q]
            w = wq[q]

            N1 = 1.0 - u - v
            N2 = u
            N3 = v
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

        all_pts.append(pts_e)
        all_phi.append(phi_e)
        all_dphix.append(dphix_e)
        all_dphiy.append(dphiy_e)
        all_detj.append(np.full((nq,), detJ, dtype=np.float64))
        all_n2.append(n2_e)
        all_nodes.append(np.tile(tri[None, :], (nq, 1)))

    all_pts = np.concatenate(all_pts, axis=0)
    all_phi = np.concatenate(all_phi, axis=0)
    all_dphix = np.concatenate(all_dphix, axis=0)
    all_dphiy = np.concatenate(all_dphiy, axis=0)
    all_detj = np.concatenate(all_detj, axis=0)
    all_n2 = np.concatenate(all_n2, axis=0)
    all_nodes = np.concatenate(all_nodes, axis=0)

    return (
        torch.tensor(all_pts, dtype=torch.float64, device=device),
        torch.tensor(all_phi, dtype=torch.float64, device=device),
        torch.tensor(all_dphix, dtype=torch.float64, device=device),
        torch.tensor(all_dphiy, dtype=torch.float64, device=device),
        torch.tensor(all_detj, dtype=torch.float64, device=device),
        torch.tensor(all_n2, dtype=torch.float64, device=device),
        torch.tensor(all_nodes, dtype=torch.long, device=device),
        torch.tensor(f_real, dtype=torch.float64, device=device),
        torch.tensor(f_imag, dtype=torch.float64, device=device),
        inside_count,
        total_count,
    )


# --- Boundary precompute ---
def precompute_boundary(nodes, boundary_edges, s1d, w1d, device):
    """
    Precompute edge quadrature for Robin term -int i*k0*u_s*v dGamma (g=0):
    edge points, linear basis, edge-length factor, node mapping.
    """
    b_pts = []
    b_phi = []
    b_len = []
    b_nodes = []
    b_w = []

    for e in boundary_edges:
        i, j = int(e[0]), int(e[1])
        p0 = nodes[i]
        p1 = nodes[j]
        elen = float(np.linalg.norm(p1 - p0))

        for q in range(len(w1d)):
            s = s1d[q]
            w = w1d[q]
            pq = (1.0 - s) * p0 + s * p1
            phi_q = np.array([1.0 - s, s], dtype=np.float64)

            b_pts.append(pq)
            b_phi.append(phi_q)
            b_len.append(elen)
            b_nodes.append([i, j])
            b_w.append(w)

    return (
        torch.tensor(np.array(b_pts), dtype=torch.float64, device=device),
        torch.tensor(np.array(b_phi), dtype=torch.float64, device=device),
        torch.tensor(np.array(b_len), dtype=torch.float64, device=device),
        torch.tensor(np.array(b_nodes), dtype=torch.long, device=device),
        torch.tensor(np.array(b_w), dtype=torch.float64, device=device),
    )


# --- VPINN weak-form loss ---
def vpinn_weak_loss(
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
    b_pts,
    b_phi,
    b_len,
    b_nodes,
    b_w,
    k0,
):
    """
    Weak-form residual:
    Re: int grad(u_r)·grad(phi) - int k0^2 n^2 u_r phi + int_Gamma k0 u_i phi - int f_r phi
    Im: int grad(u_i)·grad(phi) - int k0^2 n^2 u_i phi - int_Gamma k0 u_r phi - int f_i phi
    """
    x = vol_pts[:, 0:1].clone().detach().requires_grad_(True)
    y = vol_pts[:, 1:2].clone().detach().requires_grad_(True)

    u_re, u_im = model(x, y)

    u_re_x = grad(u_re, x, grad_outputs=torch.ones_like(u_re), create_graph=True)[0]
    u_re_y = grad(u_re, y, grad_outputs=torch.ones_like(u_re), create_graph=True)[0]
    u_im_x = grad(u_im, x, grad_outputs=torch.ones_like(u_im), create_graph=True)[0]
    u_im_y = grad(u_im, y, grad_outputs=torch.ones_like(u_im), create_graph=True)[0]

    nq = len(tri_w)
    ne = vol_pts.shape[0] // nq
    tri_w_t = torch.tensor(tri_w, dtype=torch.float64, device=vol_pts.device)
    w_flat = tri_w_t.repeat(ne)
    vol_fac = (w_flat * vol_detj)[:, None]

    grad_term_re = (u_re_x * vol_dphix + u_re_y * vol_dphiy) * vol_fac
    grad_term_im = (u_im_x * vol_dphix + u_im_y * vol_dphiy) * vol_fac
    mass_term_re = -k0 ** 2 * vol_n2[:, None] * u_re * vol_phi * vol_fac
    mass_term_im = -k0 ** 2 * vol_n2[:, None] * u_im * vol_phi * vol_fac

    contrib_re = grad_term_re + mass_term_re
    contrib_im = grad_term_im + mass_term_im

    nnode = f_real.shape[0]
    lhs_re = torch.zeros(nnode, dtype=torch.float64, device=vol_pts.device)
    lhs_im = torch.zeros(nnode, dtype=torch.float64, device=vol_pts.device)
    lhs_re.scatter_add_(0, vol_nodes.flatten(), contrib_re.flatten())
    lhs_im.scatter_add_(0, vol_nodes.flatten(), contrib_im.flatten())

    xb = b_pts[:, 0:1]
    yb = b_pts[:, 1:2]
    ub_re, ub_im = model(xb, yb)
    bfac = (b_w * b_len)[:, None]

    robin_re = -k0 * ub_im * b_phi * bfac
    robin_im =  k0 * ub_re * b_phi * bfac

    lhs_re.scatter_add_(0, b_nodes.flatten(), robin_re.flatten())
    lhs_im.scatter_add_(0, b_nodes.flatten(), robin_im.flatten())

    res_re = lhs_re - f_real
    res_im = lhs_im - f_imag

    denom = torch.mean(f_real ** 2 + f_imag ** 2) + 1e-14
    loss = torch.mean(res_re ** 2 + res_im ** 2) / denom
    return loss, res_re, res_im


# --- Training ---
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
    b_pts,
    b_phi,
    b_len,
    b_nodes,
    b_w,
    k0,
    epochs=12000,
    lr=1e-3,
):
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=600,
        min_lr=1e-6,
    )

    history = {"epoch": [], "loss": [], "lr": []}

    for ep in range(epochs):
        optimizer.zero_grad()

        loss, res_re, res_im = vpinn_weak_loss(
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
            b_pts,
            b_phi,
            b_len,
            b_nodes,
            b_w,
            k0,
        )

        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())

        if ep % 10 == 0:
            history["epoch"].append(ep)
            history["loss"].append(loss.item())
            history["lr"].append(optimizer.param_groups[0]["lr"])
            if ep % 1000 == 0:
                rms_re = torch.sqrt(torch.mean(res_re ** 2)).item()
                rms_im = torch.sqrt(torch.mean(res_im ** 2)).item()
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {ep:6d}: Loss={loss.item():.4e}, "
                    f"RMS(Re)={rms_re:.3e}, RMS(Im)={rms_im:.3e}, lr={lr_now:.2e}"
                )

    print("Training complete")
    print(f"Total training time: {time.time() - start_time:.2f} s")
    return history


# --- Visualization ---
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
    im = plt.imshow(
        normE.T,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="jet",
    )
    plt.title("normE (|E_z,total|)")
    plt.xlabel("x (um)")
    plt.ylabel("y (um)")
    plt.colorbar(im)
    plt.tight_layout()
    result_path = "SBC_weak_form_normE.png"
    plt.savefig(result_path, dpi=300, bbox_inches="tight")
    print(f"Result plot saved: {result_path}")
    plt.show()
    plt.close()


def visualize_loss_curve(history, save_name="SBC_weak_loss_curve.png"):
    if len(history["epoch"]) == 0:
        return
    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["loss"], label="weak_form_loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SBC classic loss curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    print(f"Loss curve saved: {save_name}")
    plt.show()
    plt.close()


@torch.no_grad()
def save_E_pred_grid(
    model,
    nodes,
    wavelength=1.55,
    grid_n=128,
    out_dir="SBC weak form",
    mat_name="E_pred_weaK.mat",
):
    """
    Sample |E_total| on a regular grid; E_total = E_sc + E_inc (same as SBC classic).
    """
    device = next(model.parameters()).device
    k0 = 2.0 * np.pi / wavelength

    x_min, x_max = float(nodes[:, 0].min()), float(nodes[:, 0].max())
    y_min, y_max = float(nodes[:, 1].min()), float(nodes[:, 1].max())

    gx = torch.linspace(x_min, x_max, grid_n, dtype=torch.float64, device=device)
    gy = torch.linspace(y_min, y_max, grid_n, dtype=torch.float64, device=device)
    X, Y = torch.meshgrid(gx, gy, indexing="ij")
    Xf = X.reshape(-1, 1)
    Yf = Y.reshape(-1, 1)

    model.eval()
    us_re, us_im = model(Xf, Yf)
    inc_re, inc_im = incident_field(Xf, k0)
    ut_re = us_re + inc_re
    ut_im = us_im + inc_im
    e_pred = torch.sqrt(ut_re**2 + ut_im**2).reshape(grid_n, grid_n).cpu().numpy()

    base = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base, out_dir)
    os.makedirs(save_dir, exist_ok=True)
    mat_path = os.path.join(save_dir, mat_name)
    sio.savemat(
        mat_path,
        {
            "E_pred": e_pred,
            "grid_n": np.array(grid_n, dtype=np.int64),
            "x_min": np.array(x_min, dtype=np.float64),
            "x_max": np.array(x_max, dtype=np.float64),
            "y_min": np.array(y_min, dtype=np.float64),
            "y_max": np.array(y_max, dtype=np.float64),
            "k0": np.array(k0, dtype=np.float64),
        },
    )
    print(f"E_pred ({grid_n}x{grid_n}) saved to: {mat_path}")
    model.train()


# --- Main ---
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"k0 = {k0:.8f} 1/um")

    mesh_file = "MeshData_Robin.mat"
    if not os.path.exists(mesh_file):
        if os.path.exists("MeshData_new.mat"):
            mesh_file = "MeshData_new.mat"
        else:
            raise FileNotFoundError("MeshData_Robin.mat or MeshData_new.mat not found")

    nodes, elements = load_mesh(mesh_file)
    nodes, unit_info = normalize_nodes_to_um(nodes)
    print(f"Mesh: nodes={nodes.shape[0]}, elems={elements.shape[0]}, unit={unit_info}")
    print(
        f"BBox(um): x=[{nodes[:,0].min():.4f}, {nodes[:,0].max():.4f}], "
        f"y=[{nodes[:,1].min():.4f}, {nodes[:,1].max():.4f}]"
    )

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

    b_edges = extract_boundary_edges(elements)
    s_edge, w_edge = edge_gauss_rule_2pt()
    b_pts, b_phi, b_len, b_nodes, b_w = precompute_boundary(
        nodes,
        b_edges,
        s_edge,
        w_edge,
        device,
    )
    print(f"Boundary edges={b_edges.shape[0]}, boundary quad points={b_pts.shape[0]}")

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
        b_pts,
        b_phi,
        b_len,
        b_nodes,
        b_w,
        k0,
        epochs=epochs,
        lr=lr,
    )

    torch.save(model.state_dict(), "SBC_weak_form_model.pth")
    np.savez(
        "SBC_weak_form_history.npz",
        epoch=np.array(history["epoch"]),
        loss=np.array(history["loss"]),
        lr=np.array(history["lr"]),
    )
    visualize_loss_curve(history, save_name="SBC_weak_loss_curve.png")

    visualize_normE(model, nodes, wavelength=wavelength, resolution=250)
    save_E_pred_grid(
        model,
        nodes,
        wavelength=wavelength,
        grid_n=128,
        out_dir="SBC weak form",
        mat_name="E_pred_weak.mat",
    )


if __name__ == "__main__":
    main()
