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
from scipy.io import loadmat
try:
    import h5py
except ImportError:
    h5py = None
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
from scipy.io import savemat
import time

DATA_DIR = Path(__file__).resolve().parent.parent / "samples and post proceeding" / "mat_data"


def setup_ddp():
    """Init DDP; return (rank, world_size, local_rank, device). Non-DDP: (0, 1, 0, cuda/cpu)."""
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


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

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
    # Training (use smaller batch_size 16~32 for PDE-only loss to avoid branch collapse)
    epochs: int = 1_000
    batch_size: int = 16
    learning_rate: float = 1e-3
    step_size: int = 200  # StepLR: decay every step_size epochs
    gamma: float = 0.95  # StepLR decay factor
    max_grad_norm: float = 1.0  # Gradient clipping for stability
    print_every: int = 1  # Print every N epochs
    save_every: int = 200  # Save checkpoint every N epochs

    # Data (n = sample count, e.g. deepOnet_data_A1_100 has 100 samples)
    matpath: str = "deepOnet_data_B1_50688.mat"

    # Checkpoints / outputs
    save_dir: str = "./model_save"
    results_dir: str = "./results"
    load_file_name: str = "model_B1_size_50688_ddp_fft_epoch1000"
    save_file_name: str = "model_B1_size_50688_ddp"

    # Device / dtype
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float64  # Match .mat float64 (double)
    num_workers: int = 4
    pin_memory: bool = True


class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim=2, mapping_size=64, scale=2.0):
        super(FourierFeatureMapping, self).__init__()
        self.mapping_size = mapping_size
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = 2.0 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Modified_MLP_Block(nn.Module):
    def __init__(self, input_dim, hidden_channel, output_dim, hidden_size=6):
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
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
    


def _branch_norm2d(channels):
    """InstanceNorm2d in branch: per-sample per-channel norm, no cross-sample mixing."""
    return nn.InstanceNorm2d(channels)


def add_spatial_coord_channels(epsilon_data):
    """
    Concatenate spatial coords on complex epsilon (real/imag channels).
    epsilon_data: (B, 2, H, W) -> (B, 4, H, W) [eps_real, eps_imag, x_norm, y_norm] in [0,1].
    """
    B, _, H, W = epsilon_data.shape
    device, dtype = epsilon_data.device, epsilon_data.dtype
    j = torch.linspace(0, 1, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    i = torch.linspace(0, 1, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    return torch.cat([epsilon_data, j, i], dim=1)


class CNN_Branch_Residual(nn.Module):
    """CNN branch with residuals (InstanceNorm2d; epsilon + spatial coords)."""

    def __init__(self, in_channels=4, num_classes=128):
        super(CNN_Branch_Residual, self).__init__()

        # Initial conv layers
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            _branch_norm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            _branch_norm2d(32),
            nn.SiLU(inplace=True)
        )

        # Residual blocks
        self.res_block1 = ResidualBlock(32, 64, stride=2, norm_layer=_branch_norm2d)
        self.res_block2 = ResidualBlock(64, 128, stride=2, norm_layer=_branch_norm2d)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
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
    """Residual block (optional norm_layer; GroupNorm in branch)."""

    def __init__(self, in_channels, out_channels, stride=1, norm_layer=None):
        super(ResidualBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)

        self.silu = nn.SiLU(inplace=True)

        # Downsampling shortcut
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                norm_layer(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.silu(out)

        return out


class Lambda_MLP_Branch(nn.Module):
    """Lambda branch: MLP only, no FFT."""
    def __init__(self, lambda_input_dim, output_dim_2bi, hidden_mult=4, num_layers=3):
        super(Lambda_MLP_Branch, self).__init__()
        in_dim = lambda_input_dim
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
        out = self.mlp(lambda_in)
        return out


class MIONet(nn.Module):
    """MIONet: 3 branches (Epsilon + Ebz + Lambda) + FFM trunk; lambda branch without FFT."""
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_channel, output_dim, ffm_mapping_size=64, ffm_scale=2.0):
        super(MIONet, self).__init__()
        self.output_dim = output_dim
        self.bi = output_dim // 2   # branch output vector dim
        self.bni = output_dim // 2  # trunk output vector dim
        self.branch_net = CNN_Branch_Residual(in_channels=branch_input_dim+2, num_classes=output_dim)
        self.branch_net_ebz = CNN_Branch_Residual(in_channels=branch_input_dim+2, num_classes=output_dim)
        self.branch_net_lambda = Lambda_MLP_Branch(lambda_input_dim=1, output_dim_2bi=2 * self.bi)
        self.ffm = FourierFeatureMapping(input_dim=trunk_input_dim, mapping_size=ffm_mapping_size, scale=ffm_scale)
        ffm_output_dim = ffm_mapping_size * 2
        self.trunk_net = Modified_MLP_Block(ffm_output_dim, hidden_channel, output_dim)

    def forward(self, branch_input_epsilon, branch_input_ebz, branch_input_lambda, trunk_input):
        # Epsilon -> B11, B21
        branch_input_epsilon = add_spatial_coord_channels(branch_input_epsilon)
        branch_out_eps = self.branch_net(branch_input_epsilon)  # (B, 2*bi)
        B11 = branch_out_eps[:, :self.bi]
        B21 = branch_out_eps[:, self.bi:]
        # Ebz -> B12, B22
        branch_input_ebz = add_spatial_coord_channels(branch_input_ebz)
        branch_out_ebz = self.branch_net_ebz(branch_input_ebz)
        B12 = branch_out_ebz[:, :self.bi]
        B22 = branch_out_ebz[:, self.bi:]
        # Lambda (FFT-MLP) -> B31, B32
        branch_out_lambda = self.branch_net_lambda(branch_input_lambda)  # (B, 2*bi)
        B31 = branch_out_lambda[:, :self.bi]
        B32 = branch_out_lambda[:, self.bi:]
        # Trunk -> T1, T2
        trunk_input_ffm = self.ffm(trunk_input)
        trunk_out = self.trunk_net(trunk_input_ffm)
        T1 = trunk_out[:, :, :self.bni]
        T2 = trunk_out[:, :, self.bni:]
        # Low-rank product: s_re = (B11*B12*B31)·T1, s_im = (B21*B22*B32)·T2
        s_re = torch.einsum('bi,bni->bn', B11 * B12 * B31, T1)
        s_im = torch.einsum('bi,bni->bn', B21 * B22 * B32, T2)
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
        if world_size > 1:
            self.batch_size = max(1, self.cfg.batch_size // world_size)
        self.learning_rate = self.cfg.learning_rate
        self.matpath = str(DATA_DIR / Path(self.cfg.matpath).name)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs, eta_min=1e-5)
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
                self.batch_size,
                sampler=self.train_sampler,
                shuffle=False,
                num_workers=0,
                pin_memory=self.cfg.pin_memory,
            )
        else:
            self.train_sampler = None
            self.train_loader = DataLoader(
                self.train_set,
                self.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory
            )
        test_batch_size = min(self.batch_size, len(self.test_set))
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=0 if world_size > 1 else self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory
        )
        self.ilu_cache = {}
        

    def _raw_model(self):
        """Model for state_dict save/load (.module under DDP)."""
        return self.model.module if self.world_size > 1 else self.model

    def load_model(self):
        self._raw_model().load_state_dict(
            torch.load(self.save_dir / f'{self.load_file_name}.pth', map_location=self.device, weights_only=True)
        )

    def E_function(self, epsilon_data, ebz_data, lambda_data, coord_data):
        epsilon_data = epsilon_data.to(self.device)
        ebz_data = ebz_data.to(self.device)
        lambda_data = lambda_data.to(self.device)
        coord_data = coord_data.to(self.device)
        return self.model(epsilon_data, ebz_data, lambda_data, coord_data)
    
    def get_data_loss(self, epsilon_data, ebz_data, lambda_data, coord_data, E_true):
        E_re_pred, E_im_pred = self.E_function(epsilon_data, ebz_data, lambda_data, coord_data)
        E_re_true = E_true[:,:, 0]
        E_im_true = E_true[:,:, 1]
        data_loss = self.loss_fn(E_re_pred, E_re_true) + self.loss_fn(E_im_pred, E_im_true)
        return data_loss

    def get_fem_loss(self, indices, epsilon_data, ebz_data, lambda_data, coord_data, E_true, Ai, Aj, Av, b, coord_len):
        """indices: (B,) global sample indices for ilu_cache lookup."""
        Ere_pred, Eim_pred = self.E_function(epsilon_data, ebz_data, lambda_data, coord_data)
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

        # Per-sample ILU from cache: one ILU per sample
        z_parts = []
        for i in range(B):
            start = int(offsets[i].item())
            m = int(Mi[i].item())
            r_i = r[start : start + m]
            ilu = self._get_ilu_for_global_index(int(indices[i].item()))
            z_i = ILUApply.apply(r_i, ilu)
            z_parts.append(z_i)
        z = torch.cat(z_parts, dim=0)

        loss = (z.abs() ** 2).mean()
        return loss
    
    
    @torch.no_grad()
    def test_E_loss(self):
        self.model.eval()
        total_loss = 0.0
        for indices, epsilon_data, coord_data, E_true, ebz_data, lambda_data, Ai, Aj, Av, b, coord_len in self.test_loader:
            indices = indices.to(self.device)
            epsilon_data = epsilon_data.to(self.device)
            coord_data = coord_data.to(self.device)
            ebz_data = ebz_data.to(self.device)
            lambda_data = lambda_data.to(self.device)
            loss = self.get_fem_loss(indices, epsilon_data, ebz_data, lambda_data, coord_data, E_true, Ai, Aj, Av, b, coord_len)
            total_loss += loss.item()
        self.model.train()
        return total_loss / len(self.test_loader)
    
    def train(self, epochs, print_every=100, save_every=10000):
        self.losses.append(['epoch', 'fem_loss', 'test_loss'])
        start_time = datetime.datetime.now()
        for epoch in tqdm(range(epochs), desc='Training', disable=not self.is_main):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            self.model.train()
            total_loss = 0.0
            fem_loss = 0.0

            for indices, epsilon_data, coord_data, E_true, ebz_data, lambda_data, Ai, Aj, Av, b, coord_len in self.train_loader:
                indices = indices.to(self.device)
                epsilon_data = epsilon_data.to(self.device)
                coord_data = coord_data.to(self.device)
                ebz_data = ebz_data.to(self.device)
                lambda_data = lambda_data.to(self.device)
                self.optimizer.zero_grad()

                fem_loss = self.get_fem_loss(indices, epsilon_data, ebz_data, lambda_data, coord_data, E_true, Ai, Aj, Av, b, coord_len)

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

            if self.is_main:
                if epoch % print_every == 0:
                    print(f'Epoch {epoch}, Total Loss: {avg_total_loss}, test Loss {avg_test_loss}')
                if (epoch + 1) % save_every == 0:
                    self.save_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = self.save_dir / f'{self.save_file_name}_epoch{epoch + 1}.pth'
                    torch.save(self._raw_model().state_dict(), ckpt_path)
        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self._raw_model().state_dict(), self.save_dir / f'{self.save_file_name}.pth')
            print("Current learning rate:", self.optimizer.param_groups[0]['lr'])
            print("Training Time:", (datetime.datetime.now() - start_time).total_seconds(), "s")

    def plot_loss(self):
        if not self.is_main:
            return
        if len(self.losses) < 2:
            return
        results_path = Path(self.cfg.results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        tag = self.save_file_name
        csv_path = results_path / f"loss_log_{tag}.csv"
        pd.DataFrame(self.losses[1:], columns=self.losses[0]).to_csv(csv_path, index=False)
        txt_path = results_path / f"loss_log_{tag}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            cols = self.losses[0]
            f.write("\t".join(str(c) for c in cols) + "\n")
            for row in self.losses[1:]:
                f.write("\t".join(str(x) for x in row) + "\n")
        print(f"Saved loss log: {csv_path}, {txt_path}")

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
        plot_path = results_path / f"loss_plot2_1_{tag}.png"
        plt.savefig(plot_path)
        print(f"Saved loss plot: {plot_path}")
        plt.show()
    
    def load_dataset(self):
        """Load pre-split .mat; MIONet requires Ebz_train, Ebz_test."""
        data_set = self._load_mat_compat(self.matpath)

        Epsilon_train = self._get_required_key(data_set, ['Eplison_train', 'Epsilon_train'])
        X_train = data_set['X_train']
        Ez_train = data_set['Ez_train']

        Epsilon_test = self._get_required_key(data_set, ['Eplison_test', 'Epsilon_test'])
        X_test = data_set['X_test']
        Ez_test = data_set['Ez_test']

        Ebz_train = data_set['Ebz_train']
        Ebz_test = data_set['Ebz_test']

        # MIONet branch-3 input Lambda, shape (N, bi), bi = output_dim//2
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
            print(f"Train shapes: ε {Epsilon_train.shape}, X {X_train.shape}, Ez {Ez_train.shape}, Ebz {Ebz_train.shape}, Lambda {Lambda_train.shape}")

        Train_dataset = GetDataset(
            Epsilon_train, X_train, Ez_train, Ebz_train, Lambda_train,
            Ai_train, Aj_train, Av_train, b_train, coord_len_train,
            index_offset=0
        )
        Test_dataset = GetDataset(
            Epsilon_test, X_test, Ez_test, Ebz_test, Lambda_test,
            Ai_test, Aj_test, Av_test, b_test, coord_len_test,
            index_offset=n_train
        )
        return Train_dataset, Test_dataset

    @staticmethod
    def _matlab_complex_to_numpy(arr):
        if isinstance(arr, np.ndarray) and arr.dtype.fields is not None:
            field_names = set(arr.dtype.fields.keys())
            if 'real' in field_names and 'imag' in field_names:
                return arr['real'] + 1j * arr['imag']
            if 'r' in field_names and 'i' in field_names:
                return arr['r'] + 1j * arr['i']
        return arr

    @classmethod
    def _normalize_mat_array(cls, arr):
        arr = cls._matlab_complex_to_numpy(np.array(arr))
        if arr.ndim > 1:
            arr = np.transpose(arr, axes=tuple(range(arr.ndim - 1, -1, -1)))
        return np.ascontiguousarray(arr)

    @classmethod
    def _load_mat_v73(cls, matpath):
        if h5py is None:
            raise ImportError("h5py required for MATLAB v7.3: pip install h5py")
        data = {}
        with h5py.File(matpath, 'r') as f:
            for key in f.keys():
                data[key] = cls._normalize_mat_array(f[key][()])
        return data

    @classmethod
    def _load_mat_compat(cls, matpath):
        try:
            return loadmat(matpath)
        except NotImplementedError:
            return cls._load_mat_v73(matpath)

    @staticmethod
    def _get_required_key(data, candidates):
        for key in candidates:
            if key in data:
                return data[key]
        raise KeyError(f"Missing required keys: {candidates}")

    def _get_ilu_for_global_index(self, global_index: int):
        if global_index in self.ilu_cache:
            return self.ilu_cache[global_index]

        n_train = len(self.train_set)
        if global_index < n_train:
            ds = self.train_set
            local_index = global_index
        else:
            ds = self.test_set
            local_index = global_index - n_train

        ai = ds.Ai[local_index].detach().cpu().numpy()
        aj = ds.Aj[local_index].detach().cpu().numpy()
        av = ds.Av[local_index].detach().cpu().numpy()
        Mi = int(ds.coord_len[local_index].detach().cpu().item())
        mask = (ai > 0) & (aj > 0)
        rows = (ai[mask] - 1).astype(np.int64)
        cols = (aj[mask] - 1).astype(np.int64)
        vals = av[mask]
        A = sp.coo_matrix((vals, (rows, cols)), shape=(Mi, Mi)).tocsc()
        ilu = spilu(A)
        self.ilu_cache[global_index] = ilu
        return ilu

    @torch.no_grad()
    def saveE_pred(self):
        """Batch predict and save E_pred on rank 0 only (full data, no DDP sampler)."""
        if not self.is_main:
            return
        self.model.eval()
        infer_batch_size = 64  # Small inference batch to avoid OOM
        n_train = len(self.train_set)
        n_test = len(self.test_set)

        Mmax_train = self.train_set.coord.shape[1]
        Mmax_test = self.test_set.coord.shape[1]

        E_train_pred = np.zeros((n_train, Mmax_train), dtype=np.complex128)
        E_test_pred = np.zeros((n_test, Mmax_test), dtype=np.complex128)

        # Rank 0 infers on full data, no DistributedSampler
        train_loader_full = DataLoader(self.train_set, batch_size=infer_batch_size, shuffle=False)
        for indices, epsilon_data, coord_data, E_true, ebz_data, lambda_data, *_ in train_loader_full:
            idx = indices.cpu().numpy()
            epsilon_data = epsilon_data.to(self.device)
            coord_data = coord_data.to(self.device)
            ebz_data = ebz_data.to(self.device)
            lambda_data = lambda_data.to(self.device)
            E_real, E_imag = self.E_function(epsilon_data, ebz_data, lambda_data, coord_data)
            E_pred = torch.complex(E_real, E_imag).detach().cpu().numpy()
            M_batch = E_pred.shape[1]
            E_train_pred[idx, :M_batch] = E_pred

        test_loader_infer = DataLoader(self.test_set, batch_size=infer_batch_size, shuffle=False)
        for indices, epsilon_data, coord_data, E_true, ebz_data, lambda_data, *_ in test_loader_infer:
            idx_global = indices.cpu().numpy()
            idx_local = idx_global - n_train  # Map to 0..n_test-1
            epsilon_data = epsilon_data.to(self.device)
            coord_data = coord_data.to(self.device)
            ebz_data = ebz_data.to(self.device)
            lambda_data = lambda_data.to(self.device)
            E_real, E_imag = self.E_function(epsilon_data, ebz_data, lambda_data, coord_data)
            E_pred = torch.complex(E_real, E_imag).detach().cpu().numpy()
            M_batch = E_pred.shape[1]
            E_test_pred[idx_local, :M_batch] = E_pred

        savemat(str(DATA_DIR / "E_train_pred_size_50688_ddp.mat"), {"E_pred": E_train_pred})
        savemat(str(DATA_DIR / "E_test_pred_size_50688_ddp.mat"), {"E_pred": E_test_pred})
        self.model.train()
    
    @torch.no_grad()
    def stat_test_inference_time(
        self,
        infer_batch_size=64,
        warmup_batches=1,
        infer_dtype=torch.float32,
    ):
        """
        Time test-set forward inference; per-sample average.

        Single GPU: sum of batch forward times.
        DDP: MAX across ranks as wall-clock (parallel GPUs); avg = total / full test count.

        Args:
            infer_dtype: default float32; None uses cfg.dtype; float16/bfloat16 need CUDA.

        Returns:
            dict: total_time_sec, n_samples, avg_time_per_sample_sec, world_size, infer_dtype
        """
        self.model.eval()
        n_test = len(self.test_set)
        orig_dtype = next(self.model.parameters()).dtype
        if infer_dtype is None:
            infer_dtype = self.cfg.dtype

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

        def _cast_inputs(epsilon_data, coord_data, ebz_data, lambda_data):
            return (
                epsilon_data.to(self.device, dtype=infer_dtype, non_blocking=True),
                ebz_data.to(self.device, dtype=infer_dtype, non_blocking=True),
                lambda_data.to(self.device, dtype=infer_dtype, non_blocking=True),
                coord_data.to(self.device, dtype=infer_dtype, non_blocking=True),
            )

        def _forward(epsilon_data, coord_data, ebz_data, lambda_data):
            eps, ebz, lam, coord = _cast_inputs(
                epsilon_data, coord_data, ebz_data, lambda_data
            )
            return self.model(eps, ebz, lam, coord)

        try:
            # Warmup to exclude first CUDA allocation from timing
            for batch_idx, (_, epsilon_data, coord_data, _, ebz_data, lambda_data, *_) in enumerate(
                test_loader_infer
            ):
                _forward(epsilon_data, coord_data, ebz_data, lambda_data)
                if batch_idx + 1 >= warmup_batches:
                    break
            _sync()

            local_time = 0.0
            n_local = 0
            for _, epsilon_data, coord_data, _, ebz_data, lambda_data, *_ in test_loader_infer:
                _sync()
                t0 = time.perf_counter()
                _forward(epsilon_data, coord_data, ebz_data, lambda_data)
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
                print(f"Test samples: {n_test} (world_size={self.world_size})")
                print(f"Inference dtype: {dtype_name}, training dtype: {str(orig_dtype).replace('torch.', '')}")
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
    model = MIONet(branch_input_dim=2, trunk_input_dim=2, hidden_channel=512, output_dim=256, ffm_scale=1.58)
    model = model.double()
    pinn = PINN_maxwell(model, cfg, rank=rank, world_size=world_size, local_rank=local_rank, device=device)
    # pinn.load_model()
    pinn.train(epochs=cfg.epochs, print_every=cfg.print_every, save_every=cfg.save_every)
    pinn.plot_loss()
    # pinn.saveE_pred()
    cleanup_ddp()



