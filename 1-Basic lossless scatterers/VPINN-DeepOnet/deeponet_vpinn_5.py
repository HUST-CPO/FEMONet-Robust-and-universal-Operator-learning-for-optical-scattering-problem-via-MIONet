"""
Self-contained DeepONet + VPINN single-file version (no dependency on other project .py files).

Unlike deeponet_vpinn_2, this file (v3) uses the unified weak form from Robin weak form.py:
volume and outer-boundary Robin terms are assembled on the same node residuals (matching
Robin weak form.py), with no separate loss_bc.

deeponet_vpinn_5: same logic as deeponet_vpinn_4; main writes outputs to compare/ and saves
sample_index in prediction .mat files.
"""

import os
import time
import csv
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def incident_field(x, k0):
    return torch.cos(k0 * x), -torch.sin(k0 * x)


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
    s = np.array([0.211324865405187, 0.788675134594813], dtype=np.float64)
    w = np.array([0.5, 0.5], dtype=np.float64)
    return s, w


def extract_boundary_edges(elements):
    all_edges = np.vstack([elements[:, [0, 1]], elements[:, [1, 2]], elements[:, [2, 0]]])
    all_edges = np.sort(all_edges, axis=1)
    edges, counts = np.unique(all_edges, axis=0, return_counts=True)
    return edges[counts == 1]


def boundary_nodes_from_edges(boundary_edges, n_nodes, device):
    boundary_nodes = np.unique(boundary_edges.reshape(-1))
    mask = torch.ones(n_nodes, dtype=torch.bool, device=device)
    mask[torch.as_tensor(boundary_nodes, dtype=torch.long, device=device)] = False
    bidx = torch.as_tensor(boundary_nodes, dtype=torch.long, device=device)
    return bidx, mask


def _unpack_one_sample(i, nodes, elements, domains, esz_raw, domain_to_n2, sample_id=None):
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)
    domains = np.asarray(domains, dtype=np.int64).reshape(-1)
    if isinstance(esz_raw, np.ndarray) and np.iscomplexobj(esz_raw):
        v = np.asarray(esz_raw, dtype=np.complex128).reshape(-1)
        esz_true = np.stack([np.real(v), np.imag(v)], axis=1).astype(np.float64)
    else:
        esz_true = np.asarray(esz_raw, dtype=np.float64)
    if nodes.shape[0] == 2 and nodes.shape[1] != 2:
        nodes = nodes.T
    if elements.shape[0] == 3 and elements.shape[1] != 3:
        elements = elements.T
    if elements.min() >= 1:
        elements = elements - 1
    nodes, unit_info = normalize_nodes_to_um(nodes)
    elem_n2 = np.array([float(domain_to_n2[int(d)]) for d in domains], dtype=np.float64)
    return {
        # Sample index in the original dataset (prefer train_idx/test_idx from MAT)
        "index": int(sample_id) if sample_id is not None else int(i + 1),
        "nodes": nodes,
        "elements": elements,
        "domains": domains,
        "elem_n2": elem_n2,
        "esz_true": esz_true,
        "node_unit_info": unit_info,
    }


def _load_cell_pack(data, nodes_k, elements_k, domains_k, esz_k, domain_to_n2, sample_ids=None):
    nodes_all = data[nodes_k].reshape(-1)
    elements_all = data[elements_k].reshape(-1)
    domains_all = data[domains_k].reshape(-1)
    esz_all = data[esz_k].reshape(-1)
    n = len(nodes_all)
    if sample_ids is not None and len(sample_ids) != n:
        raise ValueError(f"{nodes_k} has {n} samples but sample_ids length is {len(sample_ids)}")
    return [
        _unpack_one_sample(
            i,
            nodes_all[i],
            elements_all[i],
            domains_all[i],
            esz_all[i],
            domain_to_n2,
            sample_id=None if sample_ids is None else int(sample_ids[i]),
        )
        for i in range(n)
    ]


def load_train_test_raw(mat_path, domain_to_n2):
    data = sio.loadmat(mat_path)
    tr = ("Nodes_train", "Elements_train", "Domains_train", "Esz_train")
    te = ("Nodes_test", "Elements_test", "Domains_test", "Esz_test")
    if all(k in data for k in tr) and all(k in data for k in te):
        train_ids = None
        test_ids = None
        if "train_idx" in data and "test_idx" in data:
            train_ids = np.asarray(data["train_idx"]).reshape(-1).astype(np.int64)
            test_ids = np.asarray(data["test_idx"]).reshape(-1).astype(np.int64)
            split_msg = "explicit train/test keys + train_idx/test_idx"
        else:
            split_msg = "explicit train/test keys"
        return (
            _load_cell_pack(data, *tr, domain_to_n2, sample_ids=train_ids),
            _load_cell_pack(data, *te, domain_to_n2, sample_ids=test_ids),
            split_msg,
        )
    raise ValueError("Nodes_train/Elements_train/Domains_train/Esz_train format not found")


def precompute_volume(nodes, elements, k0, n_bg, n_scatter, r_scatter, xi, eta, w_tri, device, elem_n2=None):
    nodes_t = torch.tensor(nodes, dtype=torch.float64, device=device)
    elems_t = torch.tensor(elements, dtype=torch.long, device=device)
    x1, y1 = nodes_t[elems_t[:, 0], 0], nodes_t[elems_t[:, 0], 1]
    x2, y2 = nodes_t[elems_t[:, 1], 0], nodes_t[elems_t[:, 1], 1]
    x3, y3 = nodes_t[elems_t[:, 2], 0], nodes_t[elems_t[:, 2], 1]
    j11, j12 = x2 - x1, x3 - x1
    j21, j22 = y2 - y1, y3 - y1
    det_j = j11 * j22 - j12 * j21
    area2 = det_j.abs()

    xi_t = torch.tensor(xi, dtype=torch.float64, device=device)[None, :]
    eta_t = torch.tensor(eta, dtype=torch.float64, device=device)[None, :]
    phi1, phi2, phi3 = 1.0 - xi_t - eta_t, xi_t, eta_t
    xq = x1[:, None] * phi1 + x2[:, None] * phi2 + x3[:, None] * phi3
    yq = y1[:, None] * phi1 + y2[:, None] * phi2 + y3[:, None] * phi3
    dphi_ref = torch.tensor([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64, device=device)

    ne, nq = elements.shape[0], len(w_tri)
    inv_jt = torch.zeros((ne, 2, 2), dtype=torch.float64, device=device)
    inv_jt[:, 0, 0] = j22 / det_j
    inv_jt[:, 0, 1] = -j12 / det_j
    inv_jt[:, 1, 0] = -j21 / det_j
    inv_jt[:, 1, 1] = j11 / det_j
    dphi_xy = torch.einsum("kj,eji->eki", dphi_ref, inv_jt)
    dphix, dphiy = dphi_xy[:, :, 0], dphi_xy[:, :, 1]

    vol_pts = torch.stack([xq, yq], dim=-1).reshape(-1, 2)
    phi = torch.stack([phi1.expand(ne, -1), phi2.expand(ne, -1), phi3.expand(ne, -1)], dim=-1)
    vol_phi = phi.reshape(-1, 3)
    vol_dphix = dphix[:, None, :].expand(-1, nq, -1).reshape(-1, 3)
    vol_dphiy = dphiy[:, None, :].expand(-1, nq, -1).reshape(-1, 3)
    vol_detj = area2[:, None].expand(-1, nq).reshape(-1)
    vol_nodes = elems_t[:, None, :].expand(-1, nq, -1).reshape(-1, 3)

    if elem_n2 is not None:
        elem_n2_t = torch.tensor(elem_n2, dtype=torch.float64, device=device)
        vol_n2 = elem_n2_t[:, None].expand(-1, nq).reshape(-1)
        inside_mask = vol_n2 > (n_bg**2 + 1e-12)
    else:
        rr = torch.sqrt(vol_pts[:, 0] ** 2 + vol_pts[:, 1] ** 2)
        inside_mask = rr <= r_scatter
        vol_n2 = torch.where(inside_mask, torch.full_like(rr, n_scatter**2), torch.full_like(rr, n_bg**2))

    inc_re, inc_im = incident_field(vol_pts[:, 0:1], k0)
    coeff = k0**2 * (vol_n2[:, None] - n_bg**2)
    f_q_re = coeff * inc_re
    f_q_im = coeff * inc_im
    n_nodes = nodes.shape[0]
    f_real = torch.zeros(n_nodes, dtype=torch.float64, device=device)
    f_imag = torch.zeros(n_nodes, dtype=torch.float64, device=device)
    w_tri_t = torch.tensor(w_tri, dtype=torch.float64, device=device)
    quad_w = (vol_detj * w_tri_t.repeat(ne)).reshape(-1, 1)
    rhs_re = vol_phi * f_q_re * quad_w
    rhs_im = vol_phi * f_q_im * quad_w
    f_real.index_add_(0, vol_nodes.reshape(-1), rhs_re.reshape(-1))
    f_imag.index_add_(0, vol_nodes.reshape(-1), rhs_im.reshape(-1))

    return vol_pts, vol_phi, vol_dphix, vol_dphiy, vol_detj, vol_n2, vol_nodes, f_real, f_imag, int(inside_mask.sum().item()), int(vol_pts.shape[0])


def precompute_boundary_for_bc(nodes, boundary_edges, s_edge, w_edge, device):
    nodes_t = torch.tensor(nodes, dtype=torch.float64, device=device)
    e = torch.tensor(boundary_edges, dtype=torch.long, device=device)
    p1, p2 = nodes_t[e[:, 0]], nodes_t[e[:, 1]]
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    s_t = torch.tensor(s_edge, dtype=torch.float64, device=device)[None, :]
    w_t = torch.tensor(w_edge, dtype=torch.float64, device=device)[None, :]
    xb = x1[:, None] * (1.0 - s_t) + x2[:, None] * s_t
    yb = y1[:, None] * (1.0 - s_t) + y2[:, None] * s_t
    tx, ty = x2 - x1, y2 - y1
    edge_len = torch.sqrt(tx * tx + ty * ty + 1e-30)
    n1x, n1y = ty / edge_len, -tx / edge_len
    midx, midy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    use_n1 = (n1x * midx + n1y * midy) > ((-n1x) * midx + (-n1y) * midy)
    nx = torch.where(use_n1, n1x, -n1x)
    ny = torch.where(use_n1, n1y, -n1y)
    bc_pts = torch.stack([xb, yb], dim=-1).reshape(-1, 2)
    bc_w = (edge_len[:, None] * w_t).reshape(-1)
    bc_nx = nx[:, None].expand(-1, len(s_edge)).reshape(-1)
    bc_ny = ny[:, None].expand(-1, len(s_edge)).reshape(-1)
    return bc_pts, bc_w, bc_nx, bc_ny


def precompute_boundary_robin_weak(nodes, boundary_edges, s_edge, w_edge, device):
    """
    Match Robin weak form.py: edge Gauss points, linear basis, edge-length factor, and node
    mapping on the outer boundary for weak-form Robin assembly (no extra loss_bc).
    """
    b_pts = []
    b_phi = []
    b_len = []
    b_nodes = []
    b_w = []
    nodes = np.asarray(nodes, dtype=np.float64)
    for e in boundary_edges:
        i, j = int(e[0]), int(e[1])
        p0 = nodes[i]
        p1 = nodes[j]
        elen = float(np.linalg.norm(p1 - p0))
        for q in range(len(w_edge)):
            s = float(s_edge[q])
            w = float(w_edge[q])
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


def compute_node_n2(nodes, elements, elem_n2):
    n_nodes = nodes.shape[0]
    acc = np.zeros(n_nodes, dtype=np.float64)
    cnt = np.zeros(n_nodes, dtype=np.float64)
    for e in range(elements.shape[0]):
        v0, v1, v2 = elements[e]
        n2e = elem_n2[e]
        acc[v0] += n2e; cnt[v0] += 1.0
        acc[v1] += n2e; cnt[v1] += 1.0
        acc[v2] += n2e; cnt[v2] += 1.0
    return acc / np.maximum(cnt, 1e-14)


def _branch_norm2d(channels):
    return nn.InstanceNorm2d(channels)


def add_spatial_coord_channels(epsilon_data):
    b, _, h, w = epsilon_data.shape
    dev, dt = epsilon_data.device, epsilon_data.dtype
    j = torch.linspace(0, 1, w, device=dev, dtype=dt).view(1, 1, 1, w).expand(b, 1, h, w)
    i = torch.linspace(0, 1, h, device=dev, dtype=dt).view(1, 1, h, 1).expand(b, 1, h, w)
    return torch.cat([epsilon_data, j, i], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class CNN_Branch_Residual(nn.Module):  # CNN branch: permittivity features
    def __init__(self, in_channels=3, num_classes=128, c0=32, c1=64, c2=128):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, c0, kernel_size=3, padding=1, bias=False),
            _branch_norm2d(c0),
            nn.ReLU(inplace=True),
            nn.Conv2d(c0, c0, kernel_size=3, padding=1, bias=False),
            _branch_norm2d(c0),
            nn.ReLU(inplace=True),
        )
        self.res_block1 = ResidualBlock(c0, c1, stride=2, norm_layer=_branch_norm2d)
        self.res_block2 = ResidualBlock(c1, c2, stride=2, norm_layer=_branch_norm2d)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c2, num_classes)

    def forward(self, x):
        x = self.initial(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


class Modified_MLP_Block(nn.Module):  # MLP trunk: coordinate features
    def __init__(self, input_dim, hidden_channel, output_dim, hidden_size=6):
        super().__init__()
        self.activation = nn.Tanh()
        self.encode_u = nn.Linear(input_dim, hidden_channel)
        self.encode_v = nn.Linear(input_dim, hidden_channel)
        self.in_layer = nn.Linear(input_dim, hidden_channel)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_channel, hidden_channel) for _ in range(hidden_size)])
        self.out = nn.Linear(hidden_channel, output_dim)

    def forward(self, x):
        u = self.activation(self.encode_u(x))
        v = self.activation(self.encode_v(x))
        h = self.activation(self.in_layer(x))
        for layer in self.hidden_layers:
            z = self.activation(layer(h))
            h = (1 - z) * u + z * v
        return self.out(h)


class DeepONet(nn.Module):
    def __init__(
        self,
        trunk_input_dim=2,
        hidden_channel=128,
        output_dim=128,
        trunk_hidden_size=6,
        branch_c0=32,
        branch_c1=64,
        branch_c2=128,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.branch_net = CNN_Branch_Residual(
            in_channels=3,
            num_classes=output_dim,
            c0=branch_c0,
            c1=branch_c1,
            c2=branch_c2,
        )
        self.trunk_net = Modified_MLP_Block(
            trunk_input_dim,
            hidden_channel,
            output_dim,
            hidden_size=trunk_hidden_size,
        )

    def forward(self, branch_input, trunk_input):
        branch_input = add_spatial_coord_channels(branch_input)
        b_out = self.branch_net(branch_input)
        t_out = self.trunk_net(trunk_input)
        b1, b2 = b_out[:, :self.output_dim // 2], b_out[:, self.output_dim // 2:]
        t1, t2 = t_out[:, :, :self.output_dim // 2], t_out[:, :, self.output_dim // 2:]
        s_re = torch.einsum("bi,bni->bn", b1, t1)
        s_im = torch.einsum("bi,bni->bn", b2, t2)
        return s_re, s_im


class DeepONetVPINNWrapper(nn.Module):
    def __init__(
        self,
        hidden_channel=128,
        output_dim=128,
        trunk_hidden_size=6,
        branch_c0=32,
        branch_c1=64,
        branch_c2=128,
    ):
        super().__init__()
        self.net = DeepONet(
            trunk_input_dim=2,
            hidden_channel=hidden_channel,
            output_dim=output_dim,
            trunk_hidden_size=trunk_hidden_size,
            branch_c0=branch_c0,
            branch_c1=branch_c1,
            branch_c2=branch_c2,
        )
    def forward(self, x, y, eps_img):
        if eps_img.dim() == 3:
            eps_img = eps_img.unsqueeze(0)
        trunk = torch.cat([x, y], dim=-1).unsqueeze(0)
        s_re, s_im = self.net(eps_img, trunk)
        return s_re.squeeze(0).unsqueeze(-1), s_im.squeeze(0).unsqueeze(-1)


class MeshSampleDatasetWithEps(Dataset):
    def __init__(self, raw_samples, eps_list, k0, n_bg, n_scatter, r_scatter, device):
        self.samples = []
        xi, eta, w_tri = tri_gauss_rule_4pt()
        s_edge, w_edge = edge_gauss_rule_2pt()
        if len(raw_samples) != len(eps_list):
            raise ValueError("raw_samples and eps_list have different lengths")

        for i, raw in enumerate(raw_samples):
            nodes = raw["nodes"]
            elements = raw["elements"]
            elem_n2 = raw["elem_n2"]
            vol = precompute_volume(nodes, elements, k0, n_bg, n_scatter, r_scatter, xi, eta, w_tri, device, elem_n2)
            vol_pts, vol_phi, vol_dphix, vol_dphiy, vol_detj, vol_n2, vol_nodes, f_real, f_imag, inside_count, total_count = vol
            b_edges = extract_boundary_edges(elements)
            b_idx, interior_mask = boundary_nodes_from_edges(b_edges, nodes.shape[0], device)
            bc_pts, bc_w, bc_nx, bc_ny = precompute_boundary_for_bc(nodes, b_edges, s_edge, w_edge, device)
            b_pts, b_phi, b_len, b_nodes, b_w = precompute_boundary_robin_weak(nodes, b_edges, s_edge, w_edge, device)
            node_n2_np = compute_node_n2(nodes, elements, elem_n2)
            eps = np.asarray(eps_list[i], dtype=np.float64)
            if eps.ndim == 2:
                eps = eps[None, ...]
            self.samples.append({
                "index": raw["index"],
                "nodes_np": nodes,
                "elements_np": elements,
                "domains_np": raw["domains"],
                "w_tri": w_tri,
                "vol_pts": vol_pts,
                "vol_phi": vol_phi,
                "vol_dphix": vol_dphix,
                "vol_dphiy": vol_dphiy,
                "vol_detj": vol_detj,
                "vol_n2": vol_n2,
                "vol_nodes": vol_nodes,
                "node_n2": torch.tensor(node_n2_np, dtype=torch.float64, device=device).reshape(-1, 1),
                "f_real": f_real,
                "f_imag": f_imag,
                "inside_count": inside_count,
                "total_count": total_count,
                "boundary_edges": b_edges,
                "boundary_node_idx": b_idx,
                "interior_mask": interior_mask,
                "bc_pts": bc_pts,
                "bc_w": bc_w,
                "bc_nx": bc_nx,
                "bc_ny": bc_ny,
                "b_pts": b_pts,
                "b_phi": b_phi,
                "b_len": b_len,
                "b_nodes": b_nodes,
                "b_w": b_w,
                "esz_coords": torch.tensor(raw["nodes"], dtype=torch.float64, device=device),
                "esz_true": torch.tensor(raw["esz_true"], dtype=torch.float64, device=device),
                "epsilon_img": torch.tensor(eps, dtype=torch.float64, device=device),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_as_list(batch):
    return batch


def vpinn_weak_loss_deeponet(model, sample, k0):
    """
    Unified Robin weak-form residual (aligned with vpinn_weak_loss in Robin weak form.py),
    no separate loss_bc. Weak form: grad + mass + boundary Robin = RHS f; f matches deeponet_vpinn_2.
    """
    eps = sample["epsilon_img"]
    x = sample["vol_pts"][:, 0:1].clone().detach().requires_grad_(True)
    y = sample["vol_pts"][:, 1:2].clone().detach().requires_grad_(True)
    u_re, u_im = model(x, y, eps)
    u_re_x = grad(u_re, x, grad_outputs=torch.ones_like(u_re), create_graph=True)[0]
    u_re_y = grad(u_re, y, grad_outputs=torch.ones_like(u_re), create_graph=True)[0]
    u_im_x = grad(u_im, x, grad_outputs=torch.ones_like(u_im), create_graph=True)[0]
    u_im_y = grad(u_im, y, grad_outputs=torch.ones_like(u_im), create_graph=True)[0]

    tri_w = sample["w_tri"]
    q = len(tri_w)
    ne = sample["vol_nodes"].shape[0] // q
    w_flat = torch.tensor(tri_w, dtype=torch.float64, device=sample["vol_pts"].device).repeat(ne)
    vol_fac = (sample["vol_detj"] * w_flat).unsqueeze(-1)

    grad_term_re = (u_re_x * sample["vol_dphix"] + u_re_y * sample["vol_dphiy"]) * vol_fac
    grad_term_im = (u_im_x * sample["vol_dphix"] + u_im_y * sample["vol_dphiy"]) * vol_fac
    mass_term_re = -(k0**2) * sample["vol_n2"].unsqueeze(-1) * u_re * sample["vol_phi"] * vol_fac
    mass_term_im = -(k0**2) * sample["vol_n2"].unsqueeze(-1) * u_im * sample["vol_phi"] * vol_fac

    contrib_re = grad_term_re + mass_term_re
    contrib_im = grad_term_im + mass_term_im

    n_nodes = sample["f_real"].shape[0]
    lhs_re = torch.zeros(n_nodes, dtype=torch.float64, device=sample["vol_pts"].device)
    lhs_im = torch.zeros(n_nodes, dtype=torch.float64, device=sample["vol_pts"].device)
    lhs_re.index_add_(0, sample["vol_nodes"].reshape(-1), contrib_re.reshape(-1))
    lhs_im.index_add_(0, sample["vol_nodes"].reshape(-1), contrib_im.reshape(-1))

    b_pts = sample["b_pts"]
    xb = b_pts[:, 0:1]
    yb = b_pts[:, 1:2]
    ub_re, ub_im = model(xb, yb, eps)
    bfac = (sample["b_w"] * sample["b_len"]).unsqueeze(-1)
    robin_re = -k0 * ub_im * sample["b_phi"] * bfac
    robin_im =  k0 * ub_re * sample["b_phi"] * bfac
    lhs_re.index_add_(0, sample["b_nodes"].reshape(-1), robin_re.reshape(-1))
    lhs_im.index_add_(0, sample["b_nodes"].reshape(-1), robin_im.reshape(-1))

    res_re = lhs_re - sample["f_real"]
    res_im = lhs_im - sample["f_imag"]

    denom = torch.mean(sample["f_real"] ** 2 + sample["f_imag"] ** 2) + 1e-14
    loss_weak = torch.mean(res_re**2 + res_im**2) / denom
    return loss_weak


@torch.no_grad()
def evaluate_dataset_mse(model, dataset):
    """
    Match MATLAB mean(abs(E_true - E_nn).^2): build complex E_true/E_pred, diff = |E_true - E_pred|,
    mean(diff^2) per sample over nodes; dataset average weighted by node count (global mean of |diff|^2).
    """
    model.eval()
    total_se, total_n = 0.0, 0
    for s in dataset.samples:
        x, y = s["esz_coords"][:, 0:1], s["esz_coords"][:, 1:2]
        pr, pi = model(x, y, s["epsilon_img"])
        ez_true = torch.complex(s["esz_true"][:, 0], s["esz_true"][:, 1])
        ez_pred = torch.complex(pr.reshape(-1), pi.reshape(-1))
        diff = torch.abs(ez_true - ez_pred)
        mse = (diff**2).mean()
        n = int(s["esz_true"].shape[0])
        total_se += mse.item() * n
        total_n += n
    model.train()
    return total_se / max(total_n, 1)


def _set_log_axis_detailed(ax, ymin, ymax):
    """Log y-axis: major ticks at 10^k, minor at 2–9x, grid aligned to ticks."""
    ymin = max(float(ymin), 1e-20)
    ymax = max(float(ymax), ymin * 10)
    ax.set_yscale("log")
    ax.set_ylim(ymin * 0.35, ymax * 2.8)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=20))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="major", alpha=0.45)
    ax.grid(True, which="minor", alpha=0.22, linestyle=":")


def plot_training_loss_curves(history, save_name="deeponet_vpinn_training_curves.png"):
    """
    Training curves (per epoch): MSE_train and MSE_test on log scale.
    Function name kept for backward compatibility; plot shows MSE curves.
    """
    if len(history["epoch"]) == 0:
        return
    x = np.asarray(history["epoch"], dtype=np.float64) + 1.0
    mtr = np.maximum(np.asarray(history["mse_train"], dtype=np.float64), 1e-20)
    mte = np.maximum(np.asarray(history["mse_test"], dtype=np.float64), 1e-20)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.semilogy(x, mtr, "-s", color="#1f77b4", linewidth=1.1, markersize=3.5, label="MSE_train", zorder=3)
    ax.semilogy(x, mte, "-^", color="#ff7f0e", linewidth=1.1, markersize=3.5, label="MSE_test", zorder=3)
    ymin, ymax = float(min(mtr.min(), mte.min())), float(max(mtr.max(), mte.max()))
    _set_log_axis_detailed(ax, ymin, ymax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE |E_pred - E_true|^2 (log)")
    ax.set_title("DeepONet-VPINN: train/test MSE (complex modulus, per MATLAB)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12, integer=True, min_n_ticks=8))
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()


def plot_mse_train_test(history, save_name="deeponet_vpinn_mse_train_test.png"):
    """Legacy alias: MSE train/test curves on log scale with minor ticks."""
    if len(history["epoch"]) == 0:
        return
    mtr = np.maximum(np.asarray(history["mse_train"], dtype=np.float64), 1e-20)
    mte = np.maximum(np.asarray(history["mse_test"], dtype=np.float64), 1e-20)
    x = np.asarray(history["epoch"], dtype=np.float64) + 1.0
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.semilogy(x, mtr, "-s", label="MSE train (nodes)", linewidth=1.1, markersize=3.5)
    ax.semilogy(x, mte, "-^", label="MSE test (nodes)", linewidth=1.1, markersize=3.5)
    ymin = float(min(mtr.min(), mte.min()))
    ymax = float(max(mtr.max(), mte.max()))
    _set_log_axis_detailed(ax, ymin, ymax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE |E_pred - E_true|^2")
    ax.set_title("DeepONet-VPINN train/test MSE (complex modulus)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12, integer=True, min_n_ticks=8))
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()


def save_history_csv(history, save_path):
    """Save per-epoch training stats to a readable CSV file."""
    fields = ["epoch", "loss", "mse_train", "mse_test"]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        n = len(history["epoch"])
        for i in range(n):
            writer.writerow([
                int(history["epoch"][i]) + 1,  # display as 1-based epoch
                float(history["loss"][i]),
                float(history["mse_train"][i]),
                float(history["mse_test"][i]),
            ])
    print(f"Saved per-epoch loss CSV: {save_path}")


def train_model_batch_deeponet_vpinn(
    model,
    train_loader,
    train_ds,
    test_ds,
    k0,
    epochs=2000,
    lr=1e-2,
):
    """Train and record MSE_train / MSE_test in history every epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    history = {"epoch": [], "loss": [], "mse_train": [], "mse_test": []}
    t0 = time.time()
    for ep in range(epochs):
        ep_t0 = time.time()
        model.train()
        e_loss = 0.0
        nb = 0
        for batch in train_loader:
            optimizer.zero_grad()
            b = len(batch)
            losses = []
            t_loss = 0.0
            for s in batch:
                lw = vpinn_weak_loss_deeponet(model, s, k0)
                losses.append(lw)
                t_loss += lw.detach().item()
            e_loss += t_loss / b
            nb += 1
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            optimizer.step()

        e_loss /= max(nb, 1)
        mse_tr = evaluate_dataset_mse(model, train_ds)
        mse_te = evaluate_dataset_mse(model, test_ds)
        scheduler.step()
        history["epoch"].append(ep)
        history["loss"].append(e_loss)
        history["mse_train"].append(mse_tr)
        history["mse_test"].append(mse_te)

        elapsed = time.time() - t0
        avg_ep = elapsed / (ep + 1)
        eta = avg_ep * (epochs - ep - 1)
        em, es = divmod(elapsed, 60)
        rm, rs = divmod(eta, 60)
        print(
            f"Epoch {ep+1}/{epochs}: "
            f"MSE_train={mse_tr:.4e}, MSE_test={mse_te:.4e}, "
            f"Loss_weak={e_loss:.4e}, lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"Elapsed: {int(em)}m {int(es)}s | ETA: {int(rm)}m {int(rs)}s | EpochTime: {time.time()-ep_t0:.2f}s"
        )
    print(f"Training done in {time.time()-t0:.2f}s (history recorded every epoch, {len(history['epoch'])} points)")
    return history


@torch.no_grad()
def saveE_pred_traintest(
    model,
    train_set,
    test_set,
    train_save="E_train_pred_deeponet_vpinn.mat",
    test_save="E_test_pred_deeponet_vpinn.mat",
):
    model.eval()
    n_train = len(train_set.samples)
    n_test = len(test_set.samples)
    mmax_train = max(int(s["esz_coords"].shape[0]) for s in train_set.samples) if n_train > 0 else 0
    mmax_test = max(int(s["esz_coords"].shape[0]) for s in test_set.samples) if n_test > 0 else 0
    e_train = np.zeros((n_train, mmax_train), dtype=np.complex128)
    e_test = np.zeros((n_test, mmax_test), dtype=np.complex128)
    train_idx = np.zeros((n_train, 1), dtype=np.int64)
    test_idx = np.zeros((n_test, 1), dtype=np.int64)
    for i, s in enumerate(train_set.samples):
        train_idx[i, 0] = int(s["index"])
        x, y = s["esz_coords"][:, 0:1], s["esz_coords"][:, 1:2]
        pr, pi = model(x, y, s["epsilon_img"])
        ep = torch.complex(pr, pi).detach().cpu().numpy().reshape(-1)
        e_train[i, :ep.shape[0]] = ep
    for i, s in enumerate(test_set.samples):
        test_idx[i, 0] = int(s["index"])
        x, y = s["esz_coords"][:, 0:1], s["esz_coords"][:, 1:2]
        pr, pi = model(x, y, s["epsilon_img"])
        ep = torch.complex(pr, pi).detach().cpu().numpy().reshape(-1)
        e_test[i, :ep.shape[0]] = ep
    sio.savemat(train_save, {"E_pred": e_train, "sample_index": train_idx})
    sio.savemat(test_save, {"E_pred": e_test, "sample_index": test_idx})
    print(f"Saved train predictions: {train_save}, shape={e_train.shape}, sample_index shape={train_idx.shape}")
    print(f"Saved test predictions: {test_save}, shape={e_test.shape}, sample_index shape={test_idx.shape}")
    model.train()


@torch.no_grad()
def save_last_sample_relEz(model, dataset, save_path="last_sample_relEz.mat"):
    if len(dataset.samples) == 0:
        raise ValueError("Dataset is empty")
    model.eval()
    s = dataset.samples[-1]
    x, y = s["esz_coords"][:, 0:1], s["esz_coords"][:, 1:2]
    pr, pi = model(x, y, s["epsilon_img"])
    ez_pred = torch.complex(pr, pi).detach().cpu().numpy().reshape(-1)
    ez_true_ri = s["esz_true"].detach().cpu().numpy()
    ez_true = ez_true_ri[:, 0] + 1j * ez_true_ri[:, 1]
    abs_err = np.abs(ez_pred - ez_true)
    rel_ez = abs_err / (np.abs(ez_true) + 1e-12)
    sio.savemat(save_path, {"coords": s["esz_coords"].detach().cpu().numpy(), "Ez_pred": ez_pred, "Ez_true": ez_true, "relEz": rel_ez})
    print(f"Saved last-sample relEz: {save_path}, mean={np.mean(rel_ez):.4e}, max={np.max(rel_ez):.4e}")
    model.train()


def save_last_train_sample_E_pred(model, train_set, k0, save_path="vpinn_debug_dump.mat"):
    """
    Save last training sample for MATLAB VPINN loss reproduction.
    Exports network field and derivatives:
      volume quadrature: u_re, u_im, u_re_x, u_re_y, u_im_x, u_im_y
      boundary quadrature: ub_re, ub_im, ub_re_x, ub_re_y, ub_im_x, ub_im_y
    """
    if len(train_set.samples) == 0:
        raise ValueError("Training set is empty; cannot save last-sample prediction")
    model.eval()
    s = train_set.samples[-1]
    eps = s["epsilon_img"]

    # Node predictions (for cross-check)
    x_node = s["esz_coords"][:, 0:1]
    y_node = s["esz_coords"][:, 1:2]
    pr_node, pi_node = model(x_node, y_node, eps)
    ez_pred = torch.complex(pr_node, pi_node).detach().cpu().numpy().reshape(-1)

    # Volume quadrature points and derivatives
    x = s["vol_pts"][:, 0:1].clone().detach().requires_grad_(True)
    y = s["vol_pts"][:, 1:2].clone().detach().requires_grad_(True)
    u_re, u_im = model(x, y, eps)
    u_re_x = grad(u_re, x, grad_outputs=torch.ones_like(u_re), create_graph=False, retain_graph=True)[0]
    u_re_y = grad(u_re, y, grad_outputs=torch.ones_like(u_re), create_graph=False, retain_graph=True)[0]
    u_im_x = grad(u_im, x, grad_outputs=torch.ones_like(u_im), create_graph=False, retain_graph=True)[0]
    u_im_y = grad(u_im, y, grad_outputs=torch.ones_like(u_im), create_graph=False, retain_graph=True)[0]

    # Boundary quadrature points and derivatives
    xb = s["bc_pts"][:, 0:1].clone().detach().requires_grad_(True)
    yb = s["bc_pts"][:, 1:2].clone().detach().requires_grad_(True)
    ub_re, ub_im = model(xb, yb, eps)
    ub_re_x = grad(ub_re, xb, grad_outputs=torch.ones_like(ub_re), create_graph=False, retain_graph=True)[0]
    ub_re_y = grad(ub_re, yb, grad_outputs=torch.ones_like(ub_re), create_graph=False, retain_graph=True)[0]
    ub_im_x = grad(ub_im, xb, grad_outputs=torch.ones_like(ub_im), create_graph=False, retain_graph=True)[0]
    ub_im_y = grad(ub_im, yb, grad_outputs=torch.ones_like(ub_im), create_graph=False, retain_graph=False)[0]

    # Volume assembly intermediates for MATLAB term-by-term comparison
    tri_w = s["w_tri"]
    q = len(tri_w)
    ne = s["vol_nodes"].shape[0] // q
    w_flat = torch.tensor(tri_w, dtype=torch.float64, device=s["vol_pts"].device).repeat(ne)
    quad_weight = (s["vol_detj"] * w_flat).unsqueeze(-1)  # (Nq,1)

    int_grad_re = (u_re_x * s["vol_dphix"] + u_re_y * s["vol_dphiy"]) * quad_weight
    int_grad_im = (u_im_x * s["vol_dphix"] + u_im_y * s["vol_dphiy"]) * quad_weight
    int_mass_re = ((k0**2) * s["vol_n2"].unsqueeze(-1) * u_re * s["vol_phi"] * quad_weight)
    int_mass_im = ((k0**2) * s["vol_n2"].unsqueeze(-1) * u_im * s["vol_phi"] * quad_weight)
    local_res_re = int_grad_re - int_mass_re
    local_res_im = int_grad_im - int_mass_im

    # Python-side weak loss reference for MATLAB comparison
    py_loss_weak = vpinn_weak_loss_deeponet(model, s, k0)

    dump = {
        "nodes": s["nodes_np"],
        "elements": s["elements_np"],
        "domains": s["domains_np"],
        "elem_n2": np.asarray(s["vol_n2"].detach().cpu().numpy().reshape(ne, q)[:, 0], dtype=np.float64),
        "k0": np.array([float(k0)], dtype=np.float64),
        "n_bg": np.array([1.0], dtype=np.float64),
        "u_re": u_re.detach().cpu().numpy(),
        "u_im": u_im.detach().cpu().numpy(),
        "u_re_x": u_re_x.detach().cpu().numpy(),
        "u_re_y": u_re_y.detach().cpu().numpy(),
        "u_im_x": u_im_x.detach().cpu().numpy(),
        "u_im_y": u_im_y.detach().cpu().numpy(),
        "ub_re": ub_re.detach().cpu().numpy(),
        "ub_im": ub_im.detach().cpu().numpy(),
        "ub_re_x": ub_re_x.detach().cpu().numpy(),
        "ub_re_y": ub_re_y.detach().cpu().numpy(),
        "ub_im_x": ub_im_x.detach().cpu().numpy(),
        "ub_im_y": ub_im_y.detach().cpu().numpy(),
        "vol_pts": s["vol_pts"].detach().cpu().numpy(),
        "vol_phi": s["vol_phi"].detach().cpu().numpy(),
        "vol_dphix": s["vol_dphix"].detach().cpu().numpy(),
        "vol_dphiy": s["vol_dphiy"].detach().cpu().numpy(),
        "vol_detj": s["vol_detj"].detach().cpu().numpy(),
        "vol_nodes": s["vol_nodes"].detach().cpu().numpy(),
        "quad_weight": quad_weight.detach().cpu().numpy(),
        "int_grad_re": int_grad_re.detach().cpu().numpy(),
        "int_grad_im": int_grad_im.detach().cpu().numpy(),
        "int_mass_re": int_mass_re.detach().cpu().numpy(),
        "int_mass_im": int_mass_im.detach().cpu().numpy(),
        "local_res_re": local_res_re.detach().cpu().numpy(),
        "local_res_im": local_res_im.detach().cpu().numpy(),
        "f_real": s["f_real"].detach().cpu().numpy(),
        "f_imag": s["f_imag"].detach().cpu().numpy(),
        "bc_pts": s["bc_pts"].detach().cpu().numpy(),
        "bc_w": s["bc_w"].detach().cpu().numpy(),
        "bc_nx": s["bc_nx"].detach().cpu().numpy(),
        "bc_ny": s["bc_ny"].detach().cpu().numpy(),
        "py_loss_weak": np.array([float(py_loss_weak.detach().cpu().item())], dtype=np.float64),
        "coords": s["esz_coords"].detach().cpu().numpy(),
        "Ez_pred": ez_pred,
    }

    # Also save under legacy filename for backward compatibility
    legacy_path = "last_train_sample_E_pred_deeponet_vpinn.mat"
    if save_path != legacy_path:
        sio.savemat(legacy_path, dump)
    sio.savemat(save_path, dump)
    print(f"Saved MATLAB verification dump: {save_path}")
    model.train()


@torch.no_grad()
def plot_last_sample_field_maps(model, dataset, save_path="last_sample_fields.png"):
    if len(dataset.samples) == 0:
        raise ValueError("Dataset is empty")
    model.eval()
    s = dataset.samples[-1]
    coord = s["esz_coords"].detach().cpu().numpy()
    x = coord[:, 0]
    y = coord[:, 1]
    tri = s["elements_np"]
    pr, pi = model(s["esz_coords"][:, 0:1], s["esz_coords"][:, 1:2], s["epsilon_img"])
    ez_pred = torch.complex(pr, pi).detach().cpu().numpy().reshape(-1)
    ez_true_ri = s["esz_true"].detach().cpu().numpy()
    ez_true = ez_true_ri[:, 0] + 1j * ez_true_ri[:, 1]
    amp_true = np.abs(ez_true)
    amp_pred = np.abs(ez_pred)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    sc0 = axs[0].tricontourf(x, y, tri, amp_true, levels=64, cmap="viridis")
    axs[0].set_title("|Ez_true|")
    axs[0].set_aspect("equal", "box")
    plt.colorbar(sc0, ax=axs[0], fraction=0.046, pad=0.04)
    sc1 = axs[1].tricontourf(x, y, tri, amp_pred, levels=64, cmap="viridis")
    axs[1].set_title("|Ez_pred|")
    axs[1].set_aspect("equal", "box")
    plt.colorbar(sc1, ax=axs[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved last-sample field plot: {save_path}")
    model.train()


@torch.no_grad()
def plot_all_samples_field_maps(model, dataset, out_dir, split_name="train"):
    """
    Plot and save |Ez| vs |Ez_true| per sample as tricontourf.
    Shared color scale (vmin/vmax) within each sample for comparison.
    Output: {out_dir}/{split_name}_{idx:03d}_id{sample_index}.png
    """
    if len(dataset.samples) == 0:
        raise ValueError("Dataset is empty")
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    n = len(dataset.samples)
    for i, s in enumerate(dataset.samples):
        idx = int(s.get("index", i))
        coord = s["esz_coords"].detach().cpu().numpy()
        x = coord[:, 0]
        y = coord[:, 1]
        tri = s["elements_np"]
        pr, pi = model(s["esz_coords"][:, 0:1], s["esz_coords"][:, 1:2], s["epsilon_img"])
        ez_pred = torch.complex(pr, pi).detach().cpu().numpy().reshape(-1)
        ez_true_ri = s["esz_true"].detach().cpu().numpy()
        ez_true = ez_true_ri[:, 0] + 1j * ez_true_ri[:, 1]
        amp_true = np.abs(ez_true)
        amp_pred = np.abs(ez_pred)
        vmin = float(min(amp_true.min(), amp_pred.min()))
        vmax = float(max(amp_true.max(), amp_pred.max()))
        if vmax <= vmin:
            vmax = vmin + 1e-12

        fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
        sc0 = axs[0].tricontourf(x, y, tri, amp_true, levels=64, cmap="viridis", vmin=vmin, vmax=vmax)
        axs[0].set_title(f"|Ez_true| (sample {i}, id={idx})")
        axs[0].set_aspect("equal", "box")
        plt.colorbar(sc0, ax=axs[0], fraction=0.046, pad=0.04)
        sc1 = axs[1].tricontourf(x, y, tri, amp_pred, levels=64, cmap="viridis", vmin=vmin, vmax=vmax)
        axs[1].set_title("|Ez_pred|")
        axs[1].set_aspect("equal", "box")
        plt.colorbar(sc1, ax=axs[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        fname = os.path.join(out_dir, f"{split_name}_{i:03d}_id{idx}.png")
        plt.savefig(fname, dpi=300)
        plt.close(fig)
    print(f"Saved {n} field plots to: {out_dir}")
    model.train()


def load_train_test_eps(mat_path):
    d = sio.loadmat(mat_path)
    if "Eplison_train" not in d or "Eplison_test" not in d:
        raise KeyError("Eplison_train / Eplison_test not found")
    eps_tr = np.asarray(d["Eplison_train"])
    eps_te = np.asarray(d["Eplison_test"])
    # Support (N,H,W) or (N,1,H,W)
    if eps_tr.ndim == 4 and eps_tr.shape[1] == 1:
        eps_tr = eps_tr[:, 0, :, :]
    if eps_te.ndim == 4 and eps_te.shape[1] == 1:
        eps_te = eps_te[:, 0, :, :]
    if eps_tr.ndim != 3 or eps_te.ndim != 3:
        raise ValueError(
            f"Eplison_train/test must be (N,H,W) or (N,1,H,W); got {eps_tr.shape}, {eps_te.shape}"
        )
    return [eps_tr[i] for i in range(eps_tr.shape[0])], [eps_te[i] for i in range(eps_te.shape[0])]


def main():
    # Model hyperparameters
    # -------------------------------
    hidden_channel = 128      # trunk MLP hidden width
    trunk_hidden_size = 6     # trunk MLP depth
    output_dim = 128          # DeepONet branch/trunk output dim
    branch_c0 = 32            # branch initial channels
    branch_c1 = 64            # branch res-block 1 channels
    branch_c2 = 128           # branch res-block 2 channels

    wavelength = 1.55
    k0 = 2.0 * np.pi / wavelength
    n_bg = 1.0
    n_scatter = 1.45
    r_scatter = 0.2
    batch_size = 32
    epochs = 1000
    lr = 1e-3
    

    if torch.cuda.is_available():
        req_dev = int(os.environ.get("PINN_CUDA_DEVICE", "0"))
        vis_cnt = torch.cuda.device_count()
        if req_dev < 0 or req_dev >= vis_cnt:
            raise ValueError(f"PINN_CUDA_DEVICE={req_dev} exceeds visible GPU count ({vis_cnt})")
        device = torch.device(f"cuda:{req_dev}")
    else:
        device = torch.device("cpu")
    '''
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    '''
    scatter_mat = "Train_data_A1_all.mat"
    deep_mat = "Train_data_A1_all.mat"
    domain_to_n2 = {1: n_bg**2, 2: n_scatter**2, 3: n_scatter**2, 4: n_scatter**2, 5: n_scatter**2}

    train_raw, test_raw, split_msg = load_train_test_raw(scatter_mat, domain_to_n2)
    print(f"scattering split: {split_msg}, train={len(train_raw)}, test={len(test_raw)}")
    eps_train, eps_test = load_train_test_eps(deep_mat)
    if len(eps_train) != len(train_raw) or len(eps_test) != len(test_raw):
        raise ValueError("DeepONet epsilon sample count does not match scattering train/test split")

    train_ds = MeshSampleDatasetWithEps(train_raw, eps_train, k0, n_bg, n_scatter, r_scatter, device)
    test_ds = MeshSampleDatasetWithEps(test_raw, eps_test, k0, n_bg, n_scatter, r_scatter, device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_as_list, num_workers=0)

    model = DeepONetVPINNWrapper(
        hidden_channel=hidden_channel,
        output_dim=output_dim,
        trunk_hidden_size=trunk_hidden_size,
        branch_c0=branch_c0,
        branch_c1=branch_c1,
        branch_c2=branch_c2,
    ).double().to(device)

    compare_dir = "compare_1558prj_mse"
    os.makedirs(compare_dir, exist_ok=True)

    hist = train_model_batch_deeponet_vpinn(
        model, train_loader, train_ds, test_ds, k0, epochs=epochs, lr=lr
    )

    torch.save(model.state_dict(), os.path.join(compare_dir, "deeponet_vpinn_5_model.pth"))
    np.savez(
        os.path.join(compare_dir, "deeponet_vpinn_5_history.npz"),
        epoch=np.array(hist["epoch"]),
        loss=np.array(hist["loss"]),
        mse_train=np.array(hist["mse_train"]),
        mse_test=np.array(hist["mse_test"]),
        history_sample_every=np.array([1], dtype=np.int64),
    )
    save_history_csv(hist, os.path.join(compare_dir, "deeponet_vpinn_5_history.csv"))
    plot_training_loss_curves(hist, save_name=os.path.join(compare_dir, "deeponet_vpinn_5_training_curves.png"))
    plot_mse_train_test(hist, save_name=os.path.join(compare_dir, "deeponet_vpinn_5_mse_train_test.png"))
    saveE_pred_traintest(
        model,
        train_ds,
        test_ds,
        train_save=os.path.join(compare_dir, "E_train_pred_deeponet_vpinn_5.mat"),
        test_save=os.path.join(compare_dir, "E_test_pred_deeponet_vpinn_5.mat"),
    )
    sio.savemat(
        os.path.join(compare_dir, "train_test_sample_index_deeponet_vpinn_5.mat"),
        {
            # MATLAB-style 1xN row vectors
            "trainIdx": np.array([int(s["index"]) for s in train_ds.samples], dtype=np.int64).reshape(1, -1),
            "testIdx": np.array([int(s["index"]) for s in test_ds.samples], dtype=np.int64).reshape(1, -1),
        },
    )

    plot_all_samples_field_maps(
        model, train_ds, out_dir=os.path.join(compare_dir, "all_fields_train_deeponet_vpinn_5_1558prj"), split_name="train"
    )
    plot_all_samples_field_maps(
        model, test_ds, out_dir=os.path.join(compare_dir, "all_fields_test_deeponet_vpinn_5_1558prj"), split_name="test"
    )
    print(f"All outputs written to: {os.path.abspath(compare_dir)}")


if __name__ == "__main__":
    main()

