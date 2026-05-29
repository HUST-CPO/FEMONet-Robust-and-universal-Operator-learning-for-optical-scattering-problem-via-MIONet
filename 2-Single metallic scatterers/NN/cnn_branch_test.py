import os
import torch
import torch.distributed as dist
from torch.autograd import Function
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
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
import pandas as pd
# Plot modules
import time
import matplotlib.pyplot as plt
from scipy.interpolate import griddata  
from pathlib import Path
from scipy.io import savemat

DATA_DIR = Path(__file__).resolve().parent.parent / "samples and post proceeding" / "mat_data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    save_file_name: str = "model_B1_size_50688"

    # Device / dtype
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float64  # Match .mat float64 (double)
    num_workers: int = 4
    pin_memory: bool = True

class FourierFeatureMapping(nn.Module):
    """
    Fourier feature mapping (FFM): map (x,y) to sin/cos features for high-frequency fields.
    """
    def __init__(self, input_dim=2, mapping_size=64, scale=2.0):
        super(FourierFeatureMapping, self).__init__()
        self.mapping_size = mapping_size
        # Random Gaussian B, frozen (non-trainable)
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size) * scale, requires_grad=False)

    def forward(self, x):
        # x shape: (Batch, N, input_dim)
        # x @ B shape: (Batch, N, mapping_size)
        x_proj = 2.0 * np.pi * x @ self.B
        # Concat sin/cos; output dim = mapping_size * 2
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
        # Removed original gain = nn.init.calculate_gain(...)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming init works well for SiLU/ReLU
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                # (Xavier: nn.init.xavier_uniform_(m.weight))
                
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

class MIONet(nn.Module):
    """MIONet: dual branch (Epsilon + Ebz) + trunk with FFM coords"""
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_channel, output_dim, ffm_mapping_size=64, ffm_scale=2.0):
        super(MIONet, self).__init__()
        self.output_dim = output_dim
        self.bi = output_dim // 2   
        self.bni = output_dim // 2  

        # Epsilon and Ebz branches unchanged
        self.branch_net = CNN_Branch_Residual(in_channels=branch_input_dim+2, num_classes=output_dim)
        self.branch_net_ebz = CNN_Branch_Residual(in_channels=branch_input_dim+2, num_classes=output_dim)
        
        # Add FFM module
        self.ffm = FourierFeatureMapping(input_dim=trunk_input_dim, mapping_size=ffm_mapping_size, scale=ffm_scale)
        ffm_output_dim = ffm_mapping_size * 2  # e.g. 64 * 2 = 128
        
        # Trunk input: FFM-encoded features, not raw 2D coords
        self.trunk_net = Modified_MLP_Block(ffm_output_dim, hidden_channel, output_dim)

    def forward(self, branch_input_epsilon, branch_input_ebz, trunk_input):
        # 1. Branch forward
        branch_input_epsilon = add_spatial_coord_channels(branch_input_epsilon)
        branch_out_eps = self.branch_net(branch_input_epsilon)  
        B11 = branch_out_eps[:, :self.bi]   
        B21 = branch_out_eps[:, self.bi:]   

        branch_input_ebz = add_spatial_coord_channels(branch_input_ebz)
        branch_out_ebz = self.branch_net_ebz(branch_input_ebz)   
        B12 = branch_out_ebz[:, :self.bi]   
        B22 = branch_out_ebz[:, self.bi:]   

        # 2. Trunk forward (FFM-encoded coords)
        trunk_input_ffm = self.ffm(trunk_input)          # (B, n, ffm_output_dim)
        trunk_out = self.trunk_net(trunk_input_ffm)      # (B, n, output_dim)
        T1 = trunk_out[:, :, :self.bni]   
        T2 = trunk_out[:, :, self.bni:]   

        # 3. Low-rank product
        s_re = torch.einsum('bi,bni->bn', B11 * B12, T1)
        s_im = torch.einsum('bi,bni->bn', B21 * B22, T2)
        return s_re, s_im

class PINN_maxwell():
    def __init__(self, model, config: PINNConfig):
        self.cfg = config
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1

        self.device = torch.device(self.cfg.device)
        self.model = model.to(self.device, dtype=self.cfg.dtype)
        # Split batch across ranks under DDP to reduce per-rank memory.
        self.batch_size = self.cfg.batch_size
        if self.is_distributed:
            self.batch_size = max(1, self.cfg.batch_size // self.world_size)
        self.learning_rate = self.cfg.learning_rate
        self.matpath = str(DATA_DIR / Path(self.cfg.matpath).name)
        self.loss_fn = nn.MSELoss()
        # No weight_decay; CosineAnnealingLR only
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Keep the strategy that worked well in Fig. 3
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs, eta_min=1e-5)
        self.losses = []
        self.lamda = []
        self.save_file_name = self.cfg.save_file_name
        self.load_file_name = self.cfg.load_file_name
        self.save_dir = Path(self.cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_set, self.test_set = self.load_dataset()
        self.train_sampler = DistributedSampler(self.train_set, shuffle=True) if self.is_distributed else None
        self.test_sampler = DistributedSampler(self.test_set, shuffle=False) if self.is_distributed else None
        num_workers = 0 if self.is_distributed else self.cfg.num_workers
        self.train_loader = DataLoader(
            self.train_set,
            self.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=num_workers,
            pin_memory=self.cfg.pin_memory
        )
        test_batch_size = min(self.batch_size, len(self.test_set))
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=test_batch_size,
            shuffle=False,
            sampler=self.test_sampler,
            num_workers=num_workers,
            pin_memory=self.cfg.pin_memory
        )
        self.ilu_cache = {}
        # Lazy ILU per sample instead of full prebuild (saves memory per rank).
        

    def load_model(self):
        state = torch.load(
            self.save_dir / f'{self.load_file_name}.pth',
            map_location=self.device,
            weights_only=True
        )
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(state)
        else:
            self.model.load_state_dict(state)
        if self.is_distributed:
            dist.barrier()

    def _get_ilu_for_global_index(self, global_index: int):
        """Build and cache ILU on demand by global index from dataset."""
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

    def E_function(self, epsilon_data, ebz_data, coord_data):
        epsilon_data = epsilon_data.to(self.device, non_blocking=True)
        ebz_data = ebz_data.to(self.device, non_blocking=True)
        coord_data = coord_data.to(self.device, non_blocking=True)
        return self.model(epsilon_data, ebz_data, coord_data)
    
    def get_data_loss(self, epsilon_data, ebz_data, coord_data, E_true):
        E_re_pred, E_im_pred = self.E_function(epsilon_data, ebz_data, coord_data)
        E_re_true = E_true[:,:, 0]
        E_im_true = E_true[:,:, 1]
        data_loss = self.loss_fn(E_re_pred, E_re_true) + self.loss_fn(E_im_pred, E_im_true)
        return data_loss

    def get_fem_loss(self, indices, epsilon_data, ebz_data, coord_data, E_true, Ai, Aj, Av, b, coord_len):
        """indices: (B,) global sample indices for ilu_cache lookup."""
        Ere_pred, Eim_pred = self.E_function(epsilon_data, ebz_data, coord_data)
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
            global_idx = int(indices[i].item())
            ilu = self._get_ilu_for_global_index(global_idx)
            z_i = ILUApply.apply(r_i, ilu)
            z_parts.append(z_i)
        z = torch.cat(z_parts, dim=0)

        # Absolute loss (most effective for 0.5-scale physical fields)
        loss = (z.abs() ** 2).mean()
        
        return loss
        
    
    
    @torch.no_grad()
    def test_E_loss(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        for indices, epsilon_data, coord_data, E_true, ebz_data, Ai, Aj, Av, b, coord_len in self.test_loader:
            indices = indices.to(self.device, non_blocking=True)
            epsilon_data = epsilon_data.to(self.device, non_blocking=True)
            coord_data = coord_data.to(self.device, non_blocking=True)
            ebz_data = ebz_data.to(self.device, non_blocking=True)
            loss = self.get_fem_loss(indices, epsilon_data, ebz_data, coord_data, E_true, Ai, Aj, Av, b, coord_len)
            total_loss += loss.item()
            num_batches += 1
        if self.is_distributed:
            stats = torch.tensor([total_loss, float(num_batches)], device=self.device, dtype=torch.float64)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss = stats[0].item()
            num_batches = int(stats[1].item())
        self.model.train()
        return total_loss / max(num_batches, 1)
    
    def train(self, epochs, print_every=1, save_every=1000):
        # Log FEM loss and test loss
        self.losses.append(['epoch', 'fem_loss', 'test_loss'])
        start_time = datetime.datetime.now()
        
        for epoch in tqdm(range(epochs), desc='Training', disable=(self.rank != 0)):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            self.model.train()
            total_loss = 0.0

            for indices, epsilon_data, coord_data, E_true, ebz_data, Ai, Aj, Av, b, coord_len in self.train_loader:
                indices = indices.to(self.device, non_blocking=True)
                epsilon_data = epsilon_data.to(self.device, non_blocking=True)
                coord_data = coord_data.to(self.device, non_blocking=True)
                ebz_data = ebz_data.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()

                # FEM physics residual loss only
                fem_loss = self.get_fem_loss(indices, epsilon_data, ebz_data, coord_data, E_true, Ai, Aj, Av, b, coord_len)
                loss = fem_loss

                loss.backward()
                
                # Clip gradients during physics-only optimization
                if self.cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    
                self.optimizer.step()
                total_loss += loss.item()

            if self.is_distributed:
                train_stats = torch.tensor([total_loss, float(len(self.train_loader))], device=self.device, dtype=torch.float64)
                dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
                avg_total_loss = train_stats[0].item() / max(train_stats[1].item(), 1.0)
            else:
                avg_total_loss = total_loss / len(self.train_loader)
            avg_test_loss = self.test_E_loss() # Evaluate FEM loss on test set
            
            if self.rank == 0:
                self.losses.append([epoch, avg_total_loss, avg_test_loss])
            self.scheduler.step()

            if self.rank == 0 and epoch % print_every == 0:
                # Print loss and LR for monitoring
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch} | FEM Loss: {avg_total_loss:.5e} | Test Loss {avg_test_loss:.5e} | LR: {current_lr:.2e}')
                
            if self.rank == 0 and (epoch + 1) % save_every == 0:
                ckpt_path = self.save_dir / f'{self.save_file_name}_epoch{epoch + 1}.pth'
                state = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
                torch.save(state, ckpt_path)
                
        if self.rank == 0:
            state = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
            torch.save(state, self.save_dir / f"{self.save_file_name}.pth")
            print("Final learning rate:", self.optimizer.param_groups[0]['lr'])
            print("Training Time:", (datetime.datetime.now() - start_time).total_seconds(), "s")
        
    def plot_loss(self):
        if self.rank != 0:
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
        save_path = 'results/loss_plot2_1.png'
        plt.savefig(save_path)
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

        # MIONet branch-2 input Ebz, same shape as Epsilon (N, 2, H, W)
        Ebz_train = data_set['Ebz_train']
        Ebz_test = data_set['Ebz_test']

        coord_len_train = data_set['coord_len_train']
        coord_len_test = data_set['coord_len_test']

        Ai_train, Aj_train = data_set['Ai_train'], data_set['Aj_train']
        Ai_test, Aj_test = data_set['Ai_test'], data_set['Aj_test']
        Av_train, Av_test = data_set['Av_train'], data_set['Av_test']
        b_train, b_test = data_set['b_train'], data_set['b_test']

        n_train, n_test = len(Epsilon_train), len(Epsilon_test)
        if self.rank == 0:
            print(f"Train samples: {n_train}, Test samples: {n_test}")
            print(f"Train shapes: ε {Epsilon_train.shape}, X {X_train.shape}, Ez {Ez_train.shape}, Ebz {Ebz_train.shape}")

        # Pass Lambda=None explicitly without Lambda branch; use keywords
        Train_dataset = GetDataset(
            Epsilon_train, X_train, Ez_train,
            Ebz=Ebz_train, Lambda=None,
            Ai=Ai_train, Aj=Aj_train, Av=Av_train, b=b_train,
            coord_len=coord_len_train, index_offset=0,
        )
        Test_dataset = GetDataset(
            Epsilon_test, X_test, Ez_test,
            Ebz=Ebz_test, Lambda=None,
            Ai=Ai_test, Aj=Aj_test, Av=Av_test, b=b_test,
            coord_len=coord_len_test, index_offset=n_train,
        )
        return Train_dataset, Test_dataset

    @staticmethod
    def _matlab_complex_to_numpy(arr):
        """Convert MATLAB v7.3 complex struct to numpy complex."""
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
            # h5py MATLAB arrays are often transposed; reverse once
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

    def _build_ilu_cache(self):
        """Precompute and cache per-sample ILU for sparse A before training."""
        for idx in tqdm(range(len(self.train_set)), desc="Building ILU cache (train)", disable=(self.rank != 0)):
            ai = self.train_set.Ai[idx].numpy()
            aj = self.train_set.Aj[idx].numpy()
            av = self.train_set.Av[idx].numpy()
            Mi = int(self.train_set.coord_len[idx].item())
            mask = (ai > 0) & (aj > 0)
            rows = (ai[mask] - 1).astype(np.int64)
            cols = (aj[mask] - 1).astype(np.int64)
            vals = av[mask]
            A = sp.coo_matrix((vals, (rows, cols)), shape=(Mi, Mi)).tocsc()
            self.ilu_cache[idx] = spilu(A)
        for idx in tqdm(range(len(self.test_set)), desc="Building ILU cache (test)", disable=(self.rank != 0)):
            ai = self.test_set.Ai[idx].numpy()
            aj = self.test_set.Aj[idx].numpy()
            av = self.test_set.Av[idx].numpy()
            Mi = int(self.test_set.coord_len[idx].item())
            mask = (ai > 0) & (aj > 0)
            rows = (ai[mask] - 1).astype(np.int64)
            cols = (aj[mask] - 1).astype(np.int64)
            vals = av[mask]
            A = sp.coo_matrix((vals, (rows, cols)), shape=(Mi, Mi)).tocsc()
            self.ilu_cache[len(self.train_set) + idx] = spilu(A)
        if self.rank == 0:
            print(f"ILU cache built: {len(self.ilu_cache)} samples.")

    @torch.no_grad()
    def saveE_pred(self):
        """Batch predict/save E_pred in .mat order; small batches avoid OOM."""
        self.model.eval()
        infer_batch_size = 64  # Small inference batch to avoid OOM
        n_train = len(self.train_set)
        n_test = len(self.test_set)

        # Use coord columns (padded in .mat), match model output; not coord_len.max()
        Mmax_train = self.train_set.coord.shape[1]
        Mmax_test = self.test_set.coord.shape[1]

        E_train_pred = np.zeros((n_train, Mmax_train), dtype=np.complex128)
        E_test_pred = np.zeros((n_test, Mmax_test), dtype=np.complex128)

        # Predict train in batches; write back by indices; fill actual columns only
        for indices, epsilon_data, coord_data, E_true, ebz_data, *_ in self.train_loader:
            idx = indices.cpu().numpy()
            epsilon_data = epsilon_data.to(self.device, non_blocking=True)
            coord_data = coord_data.to(self.device, non_blocking=True)
            ebz_data = ebz_data.to(self.device, non_blocking=True)
            E_real, E_imag = self.E_function(epsilon_data, ebz_data, coord_data)
            E_pred = torch.complex(E_real, E_imag).detach().cpu().numpy()
            M_batch = E_pred.shape[1]
            E_train_pred[idx, :M_batch] = E_pred

        # Small-batch test DataLoader to avoid OOM
        test_loader_infer = DataLoader(
            self.test_set,
            batch_size=infer_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory
        )
        for indices, epsilon_data, coord_data, E_true, ebz_data, *_ in test_loader_infer:
            idx_global = indices.cpu().numpy()
            idx_local = idx_global - n_train  # Map to 0..n_test-1
            epsilon_data = epsilon_data.to(self.device, non_blocking=True)
            coord_data = coord_data.to(self.device, non_blocking=True)
            ebz_data = ebz_data.to(self.device, non_blocking=True)
            E_real, E_imag = self.E_function(epsilon_data, ebz_data, coord_data)
            E_pred = torch.complex(E_real, E_imag).detach().cpu().numpy()
            M_batch = E_pred.shape[1]
            E_test_pred[idx_local, :M_batch] = E_pred

        savemat(str(DATA_DIR / "train_pred_50688(512_256_1.58).mat"), {"E_pred": E_train_pred})
        savemat(str(DATA_DIR / "test_pred_50688(512_256_1.58).mat"), {"E_pred": E_test_pred})
        self.model.train()

    @torch.no_grad()
    def stat_test_inference_time(self, infer_batch_size=64, warmup_batches=1):
        """
        Time test forward; DDP uses max rank wall-clock over sharded inference.

        Returns:
            dict: total_time_sec, n_samples, avg_time_per_sample_sec, n_local_samples
        """
        self.model.eval()
        n_test = len(self.test_set)
        if self.is_distributed:
            infer_batch_size = max(1, infer_batch_size // self.world_size)
            num_workers = 0
        else:
            num_workers = self.cfg.num_workers

        test_loader_infer = DataLoader(
            self.test_set,
            batch_size=infer_batch_size,
            shuffle=False,
            sampler=self.test_sampler if self.is_distributed else None,
            num_workers=num_workers,
            pin_memory=self.cfg.pin_memory,
        )

        def _sync():
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            if self.is_distributed:
                dist.barrier()

        # Warmup to exclude first CUDA allocation from timing
        for batch_idx, (_, epsilon_data, coord_data, _, ebz_data, *_) in enumerate(test_loader_infer):
            epsilon_data = epsilon_data.to(self.device, non_blocking=True)
            coord_data = coord_data.to(self.device, non_blocking=True)
            ebz_data = ebz_data.to(self.device, non_blocking=True)
            self.E_function(epsilon_data, ebz_data, coord_data)
            if batch_idx + 1 >= warmup_batches:
                break
        _sync()

        total_time = 0.0
        n_local = 0
        for _, epsilon_data, coord_data, _, ebz_data, *_ in test_loader_infer:
            epsilon_data = epsilon_data.to(self.device, non_blocking=True)
            coord_data = coord_data.to(self.device, non_blocking=True)
            ebz_data = ebz_data.to(self.device, non_blocking=True)
            _sync()
            t0 = time.perf_counter()
            self.E_function(epsilon_data, ebz_data, coord_data)
            _sync()
            total_time += time.perf_counter() - t0
            n_local += epsilon_data.shape[0]

        if self.is_distributed:
            time_tensor = torch.tensor([total_time], device=self.device, dtype=torch.float64)
            dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
            total_time = time_tensor.item()

            count_tensor = torch.tensor([float(n_local)], device=self.device, dtype=torch.float64)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            n_local = int(count_tensor.item())

        avg_time = total_time / n_test if n_test > 0 else 0.0
        if self.rank == 0:
            print(f"Test samples: {n_test}")
            if self.is_distributed:
                print(f"DDP world_size: {self.world_size}, total local samples across ranks: {n_local}")
            print(f"Total inference time: {total_time:.6f} s")
            print(f"Mean time per sample: {avg_time:.6f} s ({avg_time * 1000:.3f} ms)")

        self.model.train()
        return {
            "total_time_sec": total_time,
            "n_samples": n_test,
            "n_local_samples": n_local,
            "avg_time_per_sample_sec": avg_time,
        }

if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if is_distributed:
        backend = "gloo" if os.name == "nt" else "nccl"
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
        if torch.cuda.is_available():
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            cur = torch.cuda.current_device()
            name = torch.cuda.get_device_name(cur)
            print(f"[DDP] local_rank={local_rank} current_device={cur} name={name} CUDA_VISIBLE_DEVICES={visible}", flush=True)

    cfg = PINNConfig(
    )
    if torch.cuda.is_available():
        cfg.device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        cfg.device = "cpu"
    # MIONet: dual branch (Epsilon + Ebz) + trunk, low-rank B11,B12,B21,B22 + T1,T2
    model = MIONet(branch_input_dim=2, trunk_input_dim=2, hidden_channel=512, output_dim=256,ffm_scale=1.58)
    model = model.double()
    if is_distributed and torch.cuda.is_available():
        model = model.to(torch.device(cfg.device), dtype=cfg.dtype)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    pinn = PINN_maxwell(model, cfg)
    pinn.load_model()
    #pinn.train(epochs=cfg.epochs, print_every=cfg.print_every, save_every=cfg.save_every)
    #pinn.plot_loss()
    #pinn.saveE_pred()
    pinn.stat_test_inference_time(infer_batch_size=64)
    if is_distributed:
        dist.destroy_process_group()



