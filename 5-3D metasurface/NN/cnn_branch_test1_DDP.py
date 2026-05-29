import os
import torch
from torch.autograd import Function
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# import modules
from dataclasses import dataclass
from tqdm.auto import tqdm
import numpy as np
from getdata import GetDataset
# deep learning modules
import scipy.sparse as sp
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import splu
from scipy.io import loadmat
import h5py
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import datetime
import pandas as pd
# Plot modules
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path
import csv
from scipy.io import savemat
from typing import Optional, Dict
import time

DATA_DIR = Path(__file__).resolve().parent.parent / "samples and post proceeding" / "mat_data"


def resolve_matpath(matpath: str | Path) -> Path:
    path = Path(matpath)
    if not path.is_file():
        candidate = path.with_suffix(".mat")
        if candidate.is_file():
            return candidate
    return path


def setup_ddp():
    """Init DDP; return (rank, world_size, local_rank, device). Non-distributed: (0, 1, 0, cuda/cpu)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
            dist.init_process_group(backend="nccl", device_id=device)
        else:
            device = torch.device("cpu")
            dist.init_process_group(backend="gloo")
        return rank, world_size, local_rank, device
    return 0, 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")


_INFER_DTYPE_ALIASES = {
    "float64": torch.float64,
    "double": torch.float64,
    "fp64": torch.float64,
    "float32": torch.float32,
    "float": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def parse_infer_dtype(dtype):
    """Parse string or torch.dtype to inference precision."""
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower().strip()
    if key.startswith("torch."):
        key = key.split(".", 1)[1]
    if key not in _INFER_DTYPE_ALIASES:
        raise ValueError(
            f"Unsupported inference dtype: {dtype!r}; use float64, float32, float16, or bfloat16"
        )
    return _INFER_DTYPE_ALIASES[key]


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _load_mat(path):
    """Load .mat file; supports v7.3 (HDF5) and legacy formats (h5py for v7.3)."""
    if not os.path.isfile(path) and not path.endswith(".mat"):
        path_mat = path + ".mat"
        if os.path.isfile(path_mat):
            path = path_mat
    try:
        return loadmat(path, struct_as_record=False, squeeze_me=False)
    except NotImplementedError:
        pass

    def _read_dataset(f, d):
        if isinstance(d, h5py.Dataset):
            val = d[()]
        else:
            ref = None
            for k in list(d.keys()):
                ref = d[k][()]
                break
            if ref is not None and isinstance(ref, (h5py.h5r.Reference, h5py.DatasetReference)):
                val = f[ref][()]
            else:
                val = ref
        if hasattr(val, "dtype") and getattr(val.dtype, "names", None) in (("r", "i"), ("real", "imag")):
            names = val.dtype.names
            val = val["r"] + 1j * val["i"] if "r" in names else val["real"] + 1j * val["imag"]
        val = np.asarray(val)
        if val.ndim >= 2:
            val = val.T
        return val

    with h5py.File(path, "r") as f:
        out = {}
        for key in f.keys():
            if key.startswith("#") or key.startswith("__"):
                continue
            try:
                out[key] = _read_dataset(f, f[key])
            except Exception:
                continue
        return out


class ILUApply(Function):
    @staticmethod
    def forward(ctx, r_torch, ilu):
        """
        r_torch: torch complex tensor, shape (Mi,)
        ilu: fixed SciPy spilu object
        """
        ctx.ilu = ilu
        r_np = r_torch.detach().cpu().numpy()
        z_np = ilu.solve(r_np)  # z = M^{-1} r
        z = torch.from_numpy(z_np).to(r_torch.device).to(r_torch.dtype)
        return z

    @staticmethod
    def backward(ctx, grad_out):
        """
        grad_out: dL/dz
        complex: grad_r = M^{-H} grad_out
        """
        ilu = ctx.ilu
        g_np = grad_out.detach().cpu().numpy()
        gr_np = ilu.solve(g_np, trans='H')  # conjugate-transpose solve
        grad_r = torch.from_numpy(gr_np).to(grad_out.device).to(grad_out.dtype)
        return grad_r, None

@dataclass
class PINNConfig:
    # Training (for PDE residual only, use small batch_size ~16–32 to avoid branch collapse)
    epochs: int = 1_000
    batch_size: int = 4
    learning_rate: float = 1e-3
    step_size: int = 400  # StepLR: decay every step_size epochs
    gamma: float = 0.95  # StepLR decay factor
    # CosineAnnealingLR minimum learning rate
    lr_eta_min: float = 1e-5
    max_grad_norm: float = 1.0  # Gradient clipping for stability
    print_every: int = 1  # Print every N epochs
    save_every: int = 1000  # Save checkpoint every N epochs

    # Data (n = number of samples, e.g. deepOnet_data_A1_100 = 100 samples)
    matpath: str = str(DATA_DIR / "deepOnet_data_3Dcase3_261")

    # Checkpoints
    save_dir: str = "./model_save"
    results_dir: str = "./results"
    load_file_name: str = "model_3Dcase3_size_261_ddp_v4_norm_epoch1000"
    save_file_name: str = "model_3Dcase3_size_261_ddp_lamda_fft"

    # Device / precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float64  # Align with float64 in .mat (double)
    num_workers: int = 10
    pin_memory: bool = True
    # ILU (spilu) config; None = omit arg, use scipy default
    ilu_drop_tol: Optional[float] = 1e-8
    ilu_fill_factor: Optional[float] = 20.0
    ilu_drop_rule: Optional[str] = "basic"
    ilu_permc_spec: Optional[str] = "COLAMD"
    ilu_diag_pivot_thresh: Optional[float] = None
    ilu_relax: Optional[float] = None
    ilu_panel_size: Optional[int] = None
    ilu_options: Optional[Dict] = None
    # ILU needed only for FEM loss training; set False for pure inference/benchmark
    build_ilu_cache: bool = True

class Modified_MLP_Block(nn.Module):
    def __init__(self, input_dim, hidden_channel, output_dim, hidden_size=12):
        super(Modified_MLP_Block, self).__init__()
        self.activation = nn.Tanh()
        self.encodeU = nn.Linear(input_dim, hidden_channel)
        self.encodeV = nn.Linear(input_dim, hidden_channel)
        self.In = nn.Linear(input_dim, hidden_channel)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_channel, hidden_channel) for _ in range(hidden_size)
        ])
        self.out = nn.Linear(hidden_channel, output_dim)
        self._init_weights()

    def _init_weights(self):
        torch.manual_seed(123)
        gain = nn.init.calculate_gain('tanh')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        U = self.activation(self.encodeU(x))
        V = self.activation(self.encodeV(x))
        Hidden = self.activation(self.In(x))

        for layer in self.hidden_layers:
            Z = self.activation(layer(Hidden))
            Hidden = (1 - Z) * U + Z * V

        x = self.out(Hidden)
        return x

def fourier_positional_encoding(x, num_freq):
    """
    Multi-scale Fourier positional encoding: for each component in the last dim,
    concat [sin(2^j*pi*x), cos(2^j*pi*x)], j=0..num_freq-1.
    Output dim: d * (1 + 2*num_freq).
    """
    enc = [x]
    for j in range(num_freq):
        w = (2.0 ** j) * np.pi
        enc.append(torch.sin(w * x))
        enc.append(torch.cos(w * x))
    return torch.cat(enc, dim=-1)


class FFT_MLP_Trunk(nn.Module):
    """
    Trunk: FFT (real+imag) on input, concat with multi-scale Fourier encoding, then MLP.
    """
    def __init__(self, trunk_input_dim, hidden_channel, output_dim, hidden_size=12, num_freq=4):
        super(FFT_MLP_Trunk, self).__init__()
        self.num_freq = num_freq
        in_dim_fft = 2 * trunk_input_dim
        in_dim_fourier = trunk_input_dim * (1 + 2 * num_freq)
        in_dim = in_dim_fft + in_dim_fourier
        self.mlp = Modified_MLP_Block(in_dim, hidden_channel, output_dim, hidden_size)

    def forward(self, x):
        x_fft = torch.fft.fft(x, dim=-1)
        x_fft_cat = torch.cat([x_fft.real, x_fft.imag], dim=-1)
        x_fourier = fourier_positional_encoding(x, self.num_freq)
        x_cat = torch.cat([x_fft_cat, x_fourier], dim=-1)
        return self.mlp(x_cat)



def _branch_norm3d(channels):
    """InstanceNorm3d in branch: per-sample, per-channel; avoids cross-sample mixing/collapse."""
    return nn.InstanceNorm3d(channels)


def add_spatial_coord_channels(epsilon_data):
    """
    Append normalized spatial coord channels to epsilon volume so CNN can distinguish
    same-shape structures at different positions.
    epsilon_data: (B, 2, D, H, W), complex as [real, imag] -> (B, 5, D, H, W)
    channels [eps_real, eps_imag, x_norm, y_norm, z_norm], coords in [0, 1].
    """
    B, _, D, H, W = epsilon_data.shape
    device, dtype = epsilon_data.device, epsilon_data.dtype
    x = torch.linspace(0, 1, W, device=device, dtype=dtype).view(1, 1, 1, 1, W).expand(B, 1, D, H, W)
    y = torch.linspace(0, 1, H, device=device, dtype=dtype).view(1, 1, 1, H, 1).expand(B, 1, D, H, W)
    z = torch.linspace(0, 1, D, device=device, dtype=dtype).view(1, 1, D, 1, 1).expand(B, 1, D, H, W)
    return torch.cat([epsilon_data, x, y, z], dim=1)


class CNN_Branch_Residual(nn.Module):
    """3D CNN branch with residual blocks (InstanceNorm3d; input: epsilon + spatial coords)."""

    def __init__(self, in_channels=5, num_classes=128):
        super(CNN_Branch_Residual, self).__init__()

        # Initial conv block
        self.initial = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            _branch_norm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False),
            _branch_norm3d(32),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        self.res_block1 = ResidualBlock(32, 64, stride=2, norm_layer=_branch_norm3d)
        self.res_block2 = ResidualBlock(64, 128, stride=2, norm_layer=_branch_norm3d)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Fully connected head
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.initial(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    """3D residual block (optional norm_layer)."""

    def __init__(self, in_channels, out_channels, stride=1, norm_layer=None):
        super(ResidualBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Downsample shortcut
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                norm_layer(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FFT_MLP_Branch(nn.Module):
    """FFT-MLP branch: lambda (B, lambda_dim) -> FFT -> MLP -> B31,B32 (bi-dim each). Any lambda_dim."""
    def __init__(self, lambda_input_dim, output_dim_2bi, hidden_mult=4, num_layers=3):
        super(FFT_MLP_Branch, self).__init__()
        # lambda (B, lambda_input_dim) -> FFT -> real+imag concat -> (B, 2*lambda_input_dim)
        in_dim = 2 * lambda_input_dim
        bi = output_dim_2bi // 2
        hidden = max(output_dim_2bi, hidden_mult * max(bi, 1))
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, output_dim_2bi))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, lambda_in):
        # lambda_in: (B, lambda_input_dim)
        x_fft = torch.fft.fft(lambda_in, dim=-1)  # (B, lambda_input_dim) complex
        x_cat = torch.cat([x_fft.real, x_fft.imag], dim=-1)  # (B, 2*lambda_input_dim)
        out = self.mlp(x_cat)  # (B, output_dim_2bi)
        return out


class MIONet(nn.Module):
    """MIONet: two branches (Epsilon + Lambda) + trunk; low-rank B11,B21,B31,B32 (dim bi), T1,T2 (bni); outputs s_re, s_im."""
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_channel, output_dim, trunk_num_freq=4):
        super(MIONet, self).__init__()
        self.output_dim = output_dim
        self.bi = output_dim // 2   # branch output vector dim
        self.bni = output_dim // 2  # trunk output vector dim
        self.branch_net = CNN_Branch_Residual(in_channels=branch_input_dim + 3, num_classes=output_dim)
        self.branch_net_lambda = FFT_MLP_Branch(lambda_input_dim=1, output_dim_2bi=2 * self.bi)
        self.trunk_net = FFT_MLP_Trunk(trunk_input_dim, hidden_channel, output_dim, num_freq=trunk_num_freq)

    def forward(self, branch_input_epsilon, branch_input_lambda, trunk_input):
        # Epsilon -> B11, B21
        branch_input_epsilon = add_spatial_coord_channels(branch_input_epsilon)
        branch_out_eps = self.branch_net(branch_input_epsilon)  # (B, 2*bi)
        B11 = branch_out_eps[:, :self.bi]
        B21 = branch_out_eps[:, self.bi:]
        # Lambda (FFT-MLP) -> B31, B32
        branch_out_lambda = self.branch_net_lambda(branch_input_lambda)
        B31 = branch_out_lambda[:, :self.bi]
        B32 = branch_out_lambda[:, self.bi:]
        # Trunk -> T1, T2
        trunk_out = self.trunk_net(trunk_input)
        T1 = trunk_out[:, :, :self.bni]
        T2 = trunk_out[:, :, self.bni:]
        # Low-rank product: s_re = (B11*B31)·T1, s_im = (B21*B32)·T2
        s_re = torch.einsum('bi,bni->bn', B11 * B31, T1)
        s_im = torch.einsum('bi,bni->bn', B21 * B32, T2)
        return s_re, s_im


class PINN_maxwell():
    def __init__(self, model, config: PINNConfig, rank=0, world_size=1, local_rank=0, device=None):
        self.cfg = config
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0
        self.device = device if device is not None else torch.device(self.cfg.device)

        self.model = model.to(self.device, dtype=self.cfg.dtype)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[local_rank] if self.device.type == "cuda" else None)
        self.batch_size = self.cfg.batch_size
        self.learning_rate = self.cfg.learning_rate
        self.matpath = str(resolve_matpath(self.cfg.matpath))
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Cosine annealing: LR from learning_rate down to lr_eta_min
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(int(self.cfg.epochs), 1),
            eta_min=self.cfg.lr_eta_min,
        )
        self.losses = []
        self.lamda = []
        self.save_file_name = self.cfg.save_file_name
        self.load_file_name = self.cfg.load_file_name
        self.save_dir = Path(self.cfg.save_dir)
        self.train_set, self.test_set = self.load_dataset()
        if world_size > 1:
            self.train_sampler = DistributedSampler(self.train_set, num_replicas=world_size, rank=rank, shuffle=True)
            self.train_loader = DataLoader(
                self.train_set,
                self.cfg.batch_size,
                sampler=self.train_sampler,
                shuffle=False,
                num_workers=0,
                pin_memory=self.cfg.pin_memory,
            )
        else:
            self.train_sampler = None
            self.train_loader = DataLoader(self.train_set, self.cfg.batch_size, shuffle=True)
        # Batched test inference to avoid OOM from loading full test set at once
        self.test_loader = DataLoader(self.test_set, batch_size=self.cfg.batch_size, shuffle=False)
        self.ilu_cache = {}
        if self.cfg.build_ilu_cache:
            self._build_ilu_cache()
        elif self.is_main:
            print("Skipping ILU cache build (build_ilu_cache=False).")
        

    def _raw_model(self):
        """Model for save/load state_dict (DDP: use .module)."""
        return self.model.module if self.world_size > 1 else self.model

    def load_model(self):
        self._raw_model().load_state_dict(
            torch.load(self.save_dir / f'{self.load_file_name}.pth', map_location=self.device, weights_only=True)
        )

    def E_function(self, epsilon_data, lambda_data, coord_data, infer_dtype=None):
        epsilon_data = epsilon_data.to(self.device)
        lambda_data = lambda_data.to(self.device)
        coord_data = coord_data.to(self.device)
        if infer_dtype is None:
            return self.model(epsilon_data, lambda_data, coord_data)
        infer_dtype = parse_infer_dtype(infer_dtype)
        epsilon_data = epsilon_data.to(dtype=infer_dtype)
        lambda_data = lambda_data.to(dtype=infer_dtype)
        coord_data = coord_data.to(dtype=infer_dtype)
        orig_dtype = next(self.model.parameters()).dtype
        if infer_dtype != orig_dtype:
            self.model.to(dtype=infer_dtype)
        try:
            return self.model(epsilon_data, lambda_data, coord_data)
        finally:
            if infer_dtype != orig_dtype:
                self.model.to(dtype=orig_dtype)
    
    def get_data_loss(self, epsilon_data, lambda_data, coord_data, E_true):
        E_re_pred, E_im_pred = self.E_function(epsilon_data, lambda_data, coord_data)
        E_re_true = E_true[:,:, 0]
        E_im_true = E_true[:,:, 1]
        data_loss = self.loss_fn(E_re_pred, E_re_true) + self.loss_fn(E_im_pred, E_im_true)
        return data_loss

    def get_fem_loss(self, indices, epsilon_data, lambda_data, coord_data, E_true, Ai, Aj, Av, b, coord_len):
        """indices: (B,) global sample index for ilu_cache lookup."""
        Ere_pred, Eim_pred = self.E_function(epsilon_data, lambda_data, coord_data)
        E = torch.complex(Ere_pred, Eim_pred)  # shape: (B, Mmax)

        B, Mmax = E.shape
        Mi = coord_len.squeeze(-1).long().to(self.device)  # shape: (B,)

        arangeM = torch.arange(Mmax, device=self.device)  # (Mmax,)
        mask_x = arangeM[None, :] < Mi[:, None]  # (B, Mmax)

        x_flat = E[mask_x]  # (sum Mi,)
        b_flat = b.to(self.device)[mask_x].to(x_flat.dtype)  # (sum Mi,)

        sumMi = int(Mi.sum().item())
        offsets = torch.cumsum(
            torch.cat([torch.zeros(1, device=self.device, dtype=torch.long), Mi[:-1]]),
            dim=0
        )  # (B,)

        Ai = Ai.to(self.device).long()
        Aj = Aj.to(self.device).long()
        Av = Av.to(self.device).to(x_flat.dtype)
        mask_nnz = (Ai > 0) & (Aj > 0)  # (B, Kmax); padding entries are 0

        rows = (Ai - 1 + offsets.unsqueeze(1)).masked_select(mask_nnz)
        cols = (Aj - 1 + offsets.unsqueeze(1)).masked_select(mask_nnz)
        vals = Av.masked_select(mask_nnz)

        y = torch.zeros(sumMi, dtype=x_flat.dtype, device=self.device)
        y.scatter_add_(0, rows, vals * x_flat.index_select(0, cols))
        r = y - b_flat

        # Per-sample cached ILU: one ILU per sample
        z_parts = []
        for i in range(B):
            start = int(offsets[i].item())
            m = int(Mi[i].item())
            r_i = r[start : start + m]
            ilu = self.ilu_cache[int(indices[i].item())]
            z_i = ILUApply.apply(r_i, ilu)
            z_parts.append(z_i)
        z = torch.cat(z_parts, dim=0)

        loss = (z.abs() ** 2).mean()
        return loss
    
    
    @torch.no_grad()
    def test_E_loss(self):
        self.model.eval()
        total_loss = 0.0
        total_count = 0
        for indices, epsilon_data, coord_data, E_true, lambda_data, Ai, Aj, Av, b, coord_len in self.test_loader:
            B = indices.shape[0]
            indices = indices.to(self.device)
            epsilon_data = epsilon_data.to(self.device)
            coord_data = coord_data.to(self.device)
            lambda_data = lambda_data.to(self.device)
            loss = self.get_fem_loss(indices, epsilon_data, lambda_data, coord_data, E_true, Ai, Aj, Av, b, coord_len)
            total_loss += loss.item() * B
            total_count += B
        self.model.train()
        return total_loss / total_count if total_count > 0 else 0.0
    
    def train(self, epochs, print_every=100, save_every=10000):
        self.losses.append(['epoch', 'fem_loss', 'test_loss'])
        best_test_loss = float("inf")
        start_time = datetime.datetime.now()
        loss_csv_path = None
        if self.is_main:
            results_dir = Path(self.cfg.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            loss_csv_path = results_dir / f"{self.save_file_name}_loss.csv"
            with open(loss_csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["epoch", "fem_loss", "test_loss"])
        for epoch in tqdm(range(epochs), desc='Training', disable=not self.is_main):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            self.model.train()
            total_loss = 0.0
            fem_loss = 0.0

            for indices, epsilon_data, coord_data, E_true, lambda_data, Ai, Aj, Av, b, coord_len in self.train_loader:
                indices = indices.to(self.device)
                epsilon_data = epsilon_data.to(self.device)
                coord_data = coord_data.to(self.device)
                lambda_data = lambda_data.to(self.device)
                self.optimizer.zero_grad()

                fem_loss = self.get_fem_loss(indices, epsilon_data, lambda_data, coord_data, E_true, Ai, Aj, Av, b, coord_len)

                loss = fem_loss

                loss.backward()
                if self.cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()

            avg_total_loss = total_loss / len(self.train_loader)
            avg_test_loss = self.test_E_loss()
            self.losses.append([epoch, avg_total_loss, avg_test_loss])
            self.scheduler.step()

            if self.is_main and loss_csv_path is not None:
                with open(loss_csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([epoch, avg_total_loss, avg_test_loss])

            if self.is_main:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    best_ckpt_path = self.save_dir / f'{self.save_file_name}_best.pth'
                    torch.save(self._raw_model().state_dict(), best_ckpt_path)
                if epoch % print_every == 0:
                    print(f'Epoch {epoch}, Total Loss: {avg_total_loss}, test Loss {avg_test_loss}')
                if (epoch + 1) % save_every == 0:
                    ckpt_path = self.save_dir / f'{self.save_file_name}_epoch{epoch + 1}.pth'
                    torch.save(self._raw_model().state_dict(), ckpt_path)
        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self._raw_model().state_dict(), self.save_dir / f'{self.save_file_name}.pth')
            print("Current learning rate:", self.optimizer.param_groups[0]['lr'])
            print("Training Time:", (datetime.datetime.now() - start_time).total_seconds(), "s")
            if loss_csv_path is not None:
                print(f"Loss history CSV: {loss_csv_path.resolve()}")

    def plot_loss(self):
        if not self.is_main:
            return
        data = np.array(self.losses[1:])
        epochs = data[:, 0]
        train_loss = data[:, 1]
        test_loss = data[:, 2]
        plt.figure(figsize=(10, 6))
        plt.title('Training/Test Loss')
        plt.semilogy(epochs, train_loss, label='train_loss')
        plt.semilogy(epochs, test_loss,  label='test_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        Path("results").mkdir(parents=True, exist_ok=True)
        save_path = 'results/loss_plot2_1.png'
        plt.savefig(save_path)
        plt.show()
    
    def load_dataset(self):
        """Load split .mat data; supports v7.3 (HDF5) and legacy formats."""
        data_set = _load_mat(self.matpath)

        Epsilon_train = data_set['Eplison_train']
        X_train = data_set['X_train']
        Ez_train = data_set['Ez_train']

        Epsilon_test = data_set['Eplison_test']
        X_test = data_set['X_test']
        Ez_test = data_set['Ez_test']

        Lambda_train = data_set['Lambda_train']
        Lambda_test = data_set['Lambda_test']

        coord_len_train = data_set['coord_len_train']
        coord_len_test = data_set['coord_len_test']

        Ai_train, Aj_train = data_set['Ai_train'], data_set['Aj_train']
        Ai_test, Aj_test = data_set['Ai_test'], data_set['Aj_test']
        Av_train, Av_test = data_set['Av_train'], data_set['Av_test']
        b_train, b_test = data_set['b_train'], data_set['b_test']

        n_train, n_test = len(Epsilon_train), len(Epsilon_test)
        if self.is_main:
            print(f"Train samples: {n_train}, Test samples: {n_test}")
            print(f"Train shapes: ε {Epsilon_train.shape}, X {X_train.shape}, Ez {Ez_train.shape}, Lambda {Lambda_train.shape} (2-branch: no Ebz)")

        Train_dataset = GetDataset(
            Epsilon_train, X_train, Ez_train, None, Lambda_train,
            Ai_train, Aj_train, Av_train, b_train, coord_len_train,
            index_offset=0
        )
        Test_dataset = GetDataset(
            Epsilon_test, X_test, Ez_test, None, Lambda_test,
            Ai_test, Aj_test, Av_test, b_test, coord_len_test,
            index_offset=n_train
        )
        return Train_dataset, Test_dataset

    def _build_ilu_cache(self):
        """Precompute ILU per sample before training (A fixed; compute once)."""
        ilu_kwargs = dict(
            drop_tol=self.cfg.ilu_drop_tol,
            fill_factor=self.cfg.ilu_fill_factor,
            drop_rule=self.cfg.ilu_drop_rule,
            permc_spec=self.cfg.ilu_permc_spec,
            diag_pivot_thresh=self.cfg.ilu_diag_pivot_thresh,
            relax=self.cfg.ilu_relax,
            panel_size=self.cfg.ilu_panel_size,
            options=self.cfg.ilu_options,
        )
        ilu_kwargs = {k: v for k, v in ilu_kwargs.items() if v is not None}

        for idx in tqdm(range(len(self.train_set)), desc="Building ILU cache (train)", disable=not self.is_main):
            ai = self.train_set.Ai[idx].numpy()
            aj = self.train_set.Aj[idx].numpy()
            av = self.train_set.Av[idx].numpy()
            Mi = int(self.train_set.coord_len[idx].item())
            mask = (ai > 0) & (aj > 0)
            rows = (ai[mask] - 1).astype(np.int64)
            cols = (aj[mask] - 1).astype(np.int64)
            vals = av[mask]
            A = sp.coo_matrix((vals, (rows, cols)), shape=(Mi, Mi)).tocsc()
            self.ilu_cache[idx] = spilu(A, **ilu_kwargs)
        for idx in tqdm(range(len(self.test_set)), desc="Building ILU cache (test)", disable=not self.is_main):
            ai = self.test_set.Ai[idx].numpy()
            aj = self.test_set.Aj[idx].numpy()
            av = self.test_set.Av[idx].numpy()
            Mi = int(self.test_set.coord_len[idx].item())
            mask = (ai > 0) & (aj > 0)
            rows = (ai[mask] - 1).astype(np.int64)
            cols = (aj[mask] - 1).astype(np.int64)
            vals = av[mask]
            A = sp.coo_matrix((vals, (rows, cols)), shape=(Mi, Mi)).tocsc()
            self.ilu_cache[len(self.train_set) + idx] = spilu(A, **ilu_kwargs)
        if self.is_main:
            print(f"ILU cache built: {len(self.ilu_cache)} samples.")

    @torch.no_grad()
    def saveE_pred(self):
        """Predict E in batches and save; rank 0 only (full data, no DDP sampler)."""
        if not self.is_main:
            return
        self.model.eval()
        infer_batch_size = 1  # Tiny batch at inference to limit GPU memory
        n_train = len(self.train_set)
        n_test = len(self.test_set)

        Mmax_train = self.train_set.coord.shape[1]
        Mmax_test = self.test_set.coord.shape[1]

        E_train_pred = np.zeros((n_train, Mmax_train), dtype=np.complex128)
        E_test_pred = np.zeros((n_test, Mmax_test), dtype=np.complex128)

        # Rank 0 inference on full data (no DistributedSampler)
        train_loader_full = DataLoader(self.train_set, batch_size=infer_batch_size, shuffle=False)
        for indices, epsilon_data, coord_data, E_true, lambda_data, *_ in train_loader_full:
            idx = indices.cpu().numpy()
            epsilon_data = epsilon_data.to(self.device)
            coord_data = coord_data.to(self.device)
            lambda_data = lambda_data.to(self.device)
            E_real, E_imag = self.E_function(epsilon_data, lambda_data, coord_data)
            E_pred = torch.complex(E_real, E_imag).detach().cpu().numpy()
            M_batch = E_pred.shape[1]
            E_train_pred[idx, :M_batch] = E_pred

        test_loader_infer = DataLoader(self.test_set, batch_size=infer_batch_size, shuffle=False)
        for indices, epsilon_data, coord_data, E_true, lambda_data, *_ in test_loader_infer:
            idx_global = indices.cpu().numpy()
            idx_local = idx_global - n_train  # map to 0..n_test-1
            epsilon_data = epsilon_data.to(self.device)
            coord_data = coord_data.to(self.device)
            lambda_data = lambda_data.to(self.device)
            E_real, E_imag = self.E_function(epsilon_data, lambda_data, coord_data)
            E_pred = torch.complex(E_real, E_imag).detach().cpu().numpy()
            M_batch = E_pred.shape[1]
            E_test_pred[idx_local, :M_batch] = E_pred

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        savemat(DATA_DIR / "E_train_pred_size_261_ddp.mat", {"E_pred": E_train_pred})
        savemat(DATA_DIR / "E_test_pred_size_261_ddp.mat", {"E_pred": E_test_pred})
        self.model.train()

    @torch.no_grad()
    def stat_test_inference_time(
        self,
        infer_batch_size=1,
        warmup_batches=1,
        infer_dtype=torch.float32,
    ):
        """
        Benchmark test-set forward time and per-sample average.

        Single GPU: sum of batch forward times.
        DDP: MAX across ranks as wall-clock (parallel GPUs); avg = total / full test size.

        Args:
            infer_dtype: inference precision, default float32. None uses cfg.dtype;
                torch.dtype or str e.g. "float16"/"bfloat16" (model + inputs cast).

        Returns:
            dict: total_time_sec, n_samples, avg_time_per_sample_sec, world_size, infer_dtype
        """
        self.model.eval()
        n_test = len(self.test_set)
        orig_dtype = next(self.model.parameters()).dtype
        if infer_dtype is None:
            infer_dtype = self.cfg.dtype
        else:
            infer_dtype = parse_infer_dtype(infer_dtype)
        if infer_dtype in (torch.float16, torch.bfloat16) and self.device.type == "cpu":
            raise ValueError(
                f"Inference dtype {infer_dtype} not supported on CPU; use float32/float64 or CUDA."
            )

        if self.world_size > 1:
            test_sampler = DistributedSampler(
                self.test_set,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
            test_loader_infer = DataLoader(
                self.test_set,
                batch_size=infer_batch_size,
                sampler=test_sampler,
                shuffle=False,
                num_workers=0,
                pin_memory=self.cfg.pin_memory,
            )
        else:
            test_loader_infer = DataLoader(
                self.test_set,
                batch_size=infer_batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
            )

        cast_model = infer_dtype != orig_dtype
        if cast_model:
            self.model.to(dtype=infer_dtype)

        def _sync():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        def _cast_inputs(epsilon_data, lambda_data, coord_data):
            return (
                epsilon_data.to(self.device, dtype=infer_dtype, non_blocking=True),
                lambda_data.to(self.device, dtype=infer_dtype, non_blocking=True),
                coord_data.to(self.device, dtype=infer_dtype, non_blocking=True),
            )

        def _forward(epsilon_data, lambda_data, coord_data):
            eps, lam, coord = _cast_inputs(epsilon_data, lambda_data, coord_data)
            return self.model(eps, lam, coord)

        try:
            for batch_idx, (_, epsilon_data, coord_data, _, lambda_data, *_) in enumerate(
                test_loader_infer
            ):
                _forward(epsilon_data, lambda_data, coord_data)
                if batch_idx + 1 >= warmup_batches:
                    break
            _sync()

            local_time = 0.0
            n_local = 0
            for _, epsilon_data, coord_data, _, lambda_data, *_ in test_loader_infer:
                _sync()
                t0 = time.perf_counter()
                _forward(epsilon_data, lambda_data, coord_data)
                _sync()
                local_time += time.perf_counter() - t0
                n_local += epsilon_data.size(0)

            if self.world_size > 1:
                t_tensor = torch.tensor([local_time], device=self.device, dtype=torch.float64)
                n_tensor = torch.tensor([float(n_local)], device=self.device, dtype=torch.float64)
                dist.all_reduce(t_tensor, op=dist.ReduceOp.MAX)
                dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM)
                total_time = t_tensor.item()
                n_processed = int(n_tensor.item())
            else:
                total_time = local_time
                n_processed = n_local

            avg_time = total_time / n_test if n_test > 0 else 0.0
            if self.is_main:
                dtype_name = str(infer_dtype).replace("torch.", "")
                orig_name = str(orig_dtype).replace("torch.", "")
                print(f"Test samples: {n_test} (world_size={self.world_size})")
                print(f"Inference dtype: {dtype_name}, training dtype: {orig_name}")
                if self.world_size > 1:
                    print(f"Sum of per-rank local samples (incl. sampler padding): {n_processed}")
                print(f"Total inference time: {total_time:.6f} s")
                print(f"Mean time per sample: {avg_time:.6f} s ({avg_time * 1000:.3f} ms)")

            return {
                "total_time_sec": total_time,
                "n_samples": n_test,
                "avg_time_per_sample_sec": avg_time,
                "world_size": self.world_size,
                "infer_dtype": infer_dtype,
            }
        finally:
            if cast_model:
                self.model.to(dtype=orig_dtype)
            self.model.train()    

if __name__ == "__main__":
    # Single GPU: python cnn_branch_test1_DDP.py
    # Multi-GPU DDP: torchrun --nproc_per_node=2 cnn_branch_test1_DDP.py  (set 2 to GPU count)
    rank, world_size, local_rank, device = setup_ddp()
    cfg = PINNConfig()
    model = MIONet(branch_input_dim=2, trunk_input_dim=6, hidden_channel=256, output_dim=128, trunk_num_freq=4)
    model = model.double()
    pinn = PINN_maxwell(model, cfg, rank=rank, world_size=world_size, local_rank=local_rank, device=device)
    # pinn.load_model()
    pinn.train(epochs=cfg.epochs, print_every=cfg.print_every, save_every=cfg.save_every)
    pinn.plot_loss()
    pinn.saveE_pred()
    cleanup_ddp()



