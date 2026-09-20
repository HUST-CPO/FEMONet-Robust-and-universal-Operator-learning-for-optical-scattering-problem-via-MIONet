import torch
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
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import datetime
# Plot modules
import matplotlib.pyplot as plt
from scipy.interpolate import griddata  
from pathlib import Path
from scipy.io import savemat
import time
from contextlib import contextmanager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(__file__).resolve().parent.parent / "samples and post proceeding" / "mat_data"

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
    """Parse string or torch.dtype to inference dtype."""
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


def infer_dtype_tag(dtype: torch.dtype) -> str:
    return {
        torch.float64: "float64",
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }[dtype]

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
    # Training (use smaller batch_size 16-32 for PDE residual only, to avoid branch collapse)
    epochs: int = 1_000
    batch_size: int = 8
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5  # CosineAnnealingLR minimum LR
    max_grad_norm: float = 1.0  # Gradient clipping for stable training
    print_every: int = 1  # Print every N epochs
    save_every: int = 1000  # Save checkpoint every N epochs

    # Data (n = sample count, e.g. deepOnet_data_A1_100 has 100 samples)
    matpath: str = str(DATA_DIR / "deepOnet_data_SPP14_41")

    # ILU settings (approximate A^{-1} for FEM residual accuracy)
    # Set all back to None for default ILU behavior.
    ilu_drop_tol: float | None = 1e-8
    ilu_fill_factor: float | None = 20.0
    ilu_drop_rule: str | None = "basic"
    ilu_permc_spec: str | None = "COLAMD"
    ilu_diag_pivot_thresh: float | None = None
    ilu_relax: float | None = None
    ilu_panel_size: int | None = None
    ilu_options: dict | None = None

    # Checkpoints / outputs
    save_dir: str = "./model_save"
    results_dir: str = "./results"
    load_file_name: str = "model_SPP14_size_41_epoch1000"
    save_file_name: str = "model_SPP14_size_41"

    # Device / dtype
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float64  # Match .mat float64 (double)
    num_workers: int = 10
    pin_memory: bool = True

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
    Multiscale Fourier positional encoding for SPP/high-gradient fields.
    For each x_i in last dim d, concat sin/cos at frequencies j=0..num_freq-1.
    Output dim: d * (1 + 2*num_freq). Coordinates only; loss unchanged.
    """
    # x: (..., d)
    enc = [x]
    for j in range(num_freq):
        w = 2.0 ** j * np.pi
        enc.append(torch.sin(w * x))
        enc.append(torch.cos(w * x))
    return torch.cat(enc, dim=-1)


class FFT_MLP_Trunk(nn.Module):
    """
    Trunk with FFT + multiscale Fourier encoding for SPP/high-frequency fields.
    FFT then sin/cos encoding into MLP; loss unchanged.
    """
    def __init__(self, trunk_input_dim, hidden_channel, output_dim, hidden_size=6, num_freq=4):
        super(FFT_MLP_Trunk, self).__init__()
        self.trunk_input_dim = trunk_input_dim
        self.num_freq = num_freq
        # FFT channels: 2*trunk_input_dim; Fourier: trunk_input_dim*(1+2*num_freq)
        in_dim_fft = 2 * trunk_input_dim
        in_dim_fourier = trunk_input_dim * (1 + 2 * num_freq)
        in_dim = in_dim_fft + in_dim_fourier
        self.mlp = Modified_MLP_Block(in_dim, hidden_channel, output_dim, hidden_size)

    def forward(self, x):
        # x: (B, M, trunk_input_dim), e.g. (B, M, 4) for x1,y1,x2,y2
        x_fft = torch.fft.fft(x, dim=-1)
        x_fft_cat = torch.cat([x_fft.real, x_fft.imag], dim=-1)  # (B, M, 2*trunk_input_dim)
        x_fourier = fourier_positional_encoding(x, self.num_freq)  # (B, M, trunk_input_dim*(1+2*num_freq))
        x_cat = torch.cat([x_fft_cat, x_fourier], dim=-1)
        return self.mlp(x_cat)  # (B, M, output_dim)


def _branch_norm2d(channels):
    """InstanceNorm2d in branch: per-sample per-channel norm (no cross-sample mixing)."""
    return nn.InstanceNorm2d(channels)


def add_spatial_coord_channels(epsilon_data):
    """
    Concat spatial coord channels on epsilon for position-aware CNN.
    (B,2,H,W) complex [re,im] -> (B,4,H,W) [eps_re, eps_im, x_norm, y_norm] in [0,1].
    """
    B, _, H, W = epsilon_data.shape
    device, dtype = epsilon_data.device, epsilon_data.dtype
    j = torch.linspace(0, 1, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    i = torch.linspace(0, 1, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    return torch.cat([epsilon_data, j, i], dim=1)


class CNN_Branch_Residual(nn.Module):
    """CNN branch with residuals (InstanceNorm2d; epsilon + spatial coords for position)."""

    def __init__(self, in_channels=4, num_classes=128):
        super(CNN_Branch_Residual, self).__init__()

        # Initial conv layers
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            _branch_norm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            _branch_norm2d(32),
            nn.ReLU(inplace=True)
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

        self.relu = nn.ReLU(inplace=True)

        # Downsample shortcut
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
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FFT_MLP_Branch(nn.Module):
    """FFT MLP branch: lambda (B, lambda_dim) -> FFT -> MLP outputs B31, B32 (bi dims each)."""
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
        x_cat = torch.cat([x_fft.real, x_fft.imag], dim=0)  # (B, 2*lambda_input_dim)
        out = self.mlp(x_cat)  # (B, output_dim_2bi)
        return out


class MIONet2Branch(nn.Module):
    """2 branch (Epsilon + Lambda) + trunk; low-rank B11,B21,B31,B32; outputs s_re, s_im."""
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_channel, output_dim, trunk_num_freq=4):
        super(MIONet2Branch, self).__init__()
        self.output_dim = output_dim
        self.bi = output_dim // 2
        self.bni = output_dim // 2
        self.branch_net = CNN_Branch_Residual(in_channels=branch_input_dim+2, num_classes=output_dim)
        self.branch_net_lambda = FFT_MLP_Branch(lambda_input_dim=1, output_dim_2bi=2 * self.bi)
        self.trunk_net = FFT_MLP_Trunk(trunk_input_dim, hidden_channel, output_dim, num_freq=trunk_num_freq)

    def forward(self, branch_input_epsilon, branch_input_lambda, trunk_input):
        branch_input_epsilon = add_spatial_coord_channels(branch_input_epsilon)
        branch_out_eps = self.branch_net(branch_input_epsilon)
        B11 = branch_out_eps[:, :self.bi]
        B21 = branch_out_eps[:, self.bi:]
        branch_out_lambda = self.branch_net_lambda(branch_input_lambda)
        B31 = branch_out_lambda[:, :self.bi]
        B32 = branch_out_lambda[:, self.bi:]
        trunk_out = self.trunk_net(trunk_input)
        T1 = trunk_out[:, :, :self.bni]
        T2 = trunk_out[:, :, self.bni:]
        s_re = torch.einsum('bi,bni->bn', B11 * B31, T1)
        s_im = torch.einsum('bi,bni->bn', B21 * B32, T2)
        return s_re, s_im


class PINN_maxwell():
    def __init__(self, model, config: PINNConfig):
        self.cfg = config

        self.device = torch.device(self.cfg.device)
        self.model = model.to(self.device, dtype=self.cfg.dtype)
        self.batch_size = self.cfg.batch_size
        self.learning_rate = self.cfg.learning_rate
        self.matpath = self.cfg.matpath
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.epochs,
            eta_min=self.cfg.min_learning_rate
        )
        self.losses = []
        self.lamda = []
        self.save_file_name = self.cfg.save_file_name
        self.load_file_name = self.cfg.load_file_name
        self.save_dir = Path(self.cfg.save_dir)
        self.train_set, self.test_set = self.load_dataset()
        self.train_loader = DataLoader(self.train_set, self.cfg.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=len(self.test_set), shuffle=False)
        self.ilu_cache = {}
        self._build_ilu_cache()
        

    def load_model(self):
        self.model.load_state_dict(torch.load(self.save_dir / f'{self.load_file_name}.pth', map_location=self.device, weights_only=True))

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
        with self._infer_precision(infer_dtype):
            return self.model(epsilon_data, lambda_data, coord_data)

    @contextmanager
    def _infer_precision(self, infer_dtype: torch.dtype):
        infer_dtype = parse_infer_dtype(infer_dtype)
        orig_dtype = next(self.model.parameters()).dtype
        if infer_dtype != orig_dtype:
            self.model.to(dtype=infer_dtype)
        try:
            yield infer_dtype
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
        mask_nnz = (Ai > 0) & (Aj > 0)  # (B, Kmax); padding is 0

        rows = (Ai - 1 + offsets.unsqueeze(1)).masked_select(mask_nnz)
        cols = (Aj - 1 + offsets.unsqueeze(1)).masked_select(mask_nnz)
        vals = Av.masked_select(mask_nnz)

        y = torch.zeros(sumMi, dtype=x_flat.dtype, device=self.device)
        y.scatter_add_(0, rows, vals * x_flat.index_select(0, cols))
        r = y - b_flat

        # Per-sample ILU apply for z
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
        for indices, epsilon_data, coord_data, E_true, ebz_data, lambda_data, Ai, Aj, Av, b, coord_len in self.test_loader:
            indices = indices.to(self.device)
            epsilon_data = epsilon_data.to(self.device)
            coord_data = coord_data.to(self.device)
            lambda_data = lambda_data.to(self.device)
            loss = self.get_fem_loss(indices, epsilon_data, lambda_data, coord_data, E_true, Ai, Aj, Av, b, coord_len)
            total_loss += loss.item()
        self.model.train()
        return total_loss / len(self.test_loader)
    
    def train(self, epochs, print_every=100, save_every=10000):
        self.losses.append(['epoch', 'fem_loss', 'test_loss'])
        start_time = datetime.datetime.now()
        for epoch in tqdm(range(epochs), desc='Training'):
            self.model.train()
            total_loss = 0.0
            fem_loss = 0.0

            for indices, epsilon_data, coord_data, E_true, ebz_data, lambda_data, Ai, Aj, Av, b, coord_len in self.train_loader:
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

            if epoch % print_every == 0:
                print(f'Epoch {epoch}, Total Loss: {avg_total_loss}, test Loss {avg_test_loss}')
            if (epoch + 1) % save_every == 0:
                ckpt_path = self.save_dir / f'{self.save_file_name}_epoch{epoch + 1}.pth'
                torch.save(self.model.state_dict(), ckpt_path)
        torch.save(self.model.state_dict(), self.save_dir / f'{self.save_file_name}.pth')
        print("Current learning rate:", self.optimizer.param_groups[0]['lr'])
        print("Training Time:", (datetime.datetime.now() - start_time).total_seconds(), "s")

    def plot_loss(self):
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
        """2 branch (eps + lambda); no Ebz (GetDataset Ebz=None)."""
        mat_file = Path(self.matpath)
        if mat_file.suffix != ".mat":
            mat_file = mat_file.with_suffix(".mat")
        data_set = loadmat(str(mat_file))
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
        """Precompute and cache ILU per sample A before training (A is fixed)."""
        spilu_kwargs = dict(
            drop_tol=self.cfg.ilu_drop_tol,
            fill_factor=self.cfg.ilu_fill_factor,
            drop_rule=self.cfg.ilu_drop_rule,
            permc_spec=self.cfg.ilu_permc_spec,
            diag_pivot_thresh=self.cfg.ilu_diag_pivot_thresh,
            relax=self.cfg.ilu_relax,
            panel_size=self.cfg.ilu_panel_size,
            options=self.cfg.ilu_options,
        )
        for idx in tqdm(range(len(self.train_set)), desc="Building ILU cache (train)"):
            ai = self.train_set.Ai[idx].numpy()
            aj = self.train_set.Aj[idx].numpy()
            av = self.train_set.Av[idx].numpy()
            Mi = int(self.train_set.coord_len[idx].item())
            mask = (ai > 0) & (aj > 0)
            rows = (ai[mask] - 1).astype(np.int64)
            cols = (aj[mask] - 1).astype(np.int64)
            vals = av[mask]
            A = sp.coo_matrix((vals, (rows, cols)), shape=(Mi, Mi)).tocsc()
            self.ilu_cache[idx] = spilu(A, **spilu_kwargs)
        for idx in tqdm(range(len(self.test_set)), desc="Building ILU cache (test)"):
            ai = self.test_set.Ai[idx].numpy()
            aj = self.test_set.Aj[idx].numpy()
            av = self.test_set.Av[idx].numpy()
            Mi = int(self.test_set.coord_len[idx].item())
            mask = (ai > 0) & (aj > 0)
            rows = (ai[mask] - 1).astype(np.int64)
            cols = (aj[mask] - 1).astype(np.int64)
            vals = av[mask]
            A = sp.coo_matrix((vals, (rows, cols)), shape=(Mi, Mi)).tocsc()
            self.ilu_cache[len(self.train_set) + idx] = spilu(A, **spilu_kwargs)
        print(f"ILU cache built: {len(self.ilu_cache)} samples.")

    @torch.no_grad()
    def saveE_pred(self):
        """Predict and save E_pred in batches (same order as .mat); avoid OOM."""
        self.model.eval()
        infer_batch_size = 64  # Small infer batch to avoid OOM
        n_train = len(self.train_set)
        n_test = len(self.test_set)

        # Use coord column dim (padded in .mat); do not use coord_len.max()
        Mmax_train = self.train_set.coord.shape[1]
        Mmax_test = self.test_set.coord.shape[1]

        E_train_pred = np.zeros((n_train, Mmax_train), dtype=np.complex128)
        E_test_pred = np.zeros((n_test, Mmax_test), dtype=np.complex128)

        for indices, epsilon_data, coord_data, E_true, ebz_data, lambda_data, *_ in self.train_loader:
            idx = indices.cpu().numpy()
            epsilon_data = epsilon_data.to(self.device)
            coord_data = coord_data.to(self.device)
            lambda_data = lambda_data.to(self.device)
            E_real, E_imag = self.E_function(epsilon_data, lambda_data, coord_data)
            E_pred = torch.complex(E_real, E_imag).detach().cpu().numpy()
            M_batch = E_pred.shape[1]
            E_train_pred[idx, :M_batch] = E_pred

        # Small-batch test loader to avoid OOM
        test_loader_infer = DataLoader(
            self.test_set, batch_size=infer_batch_size, shuffle=False
        )
        for indices, epsilon_data, coord_data, E_true, ebz_data, lambda_data, *_ in test_loader_infer:
            idx_global = indices.cpu().numpy()
            idx_local = idx_global - n_train
            epsilon_data = epsilon_data.to(self.device)
            coord_data = coord_data.to(self.device)
            lambda_data = lambda_data.to(self.device)
            E_real, E_imag = self.E_function(epsilon_data, lambda_data, coord_data)
            E_pred = torch.complex(E_real, E_imag).detach().cpu().numpy()
            M_batch = E_pred.shape[1]
            E_test_pred[idx_local, :M_batch] = E_pred

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        savemat(str(DATA_DIR / "E_train_pred_size_41.mat"), {"E_pred": E_train_pred})
        savemat(str(DATA_DIR / "E_test_pred_size_41.mat"), {"E_pred": E_test_pred})
        self.model.train()

    @torch.no_grad()
    def stat_test_inference_time(
        self,
        infer_batch_size=64,
        warmup_batches=1,
        infer_dtype=torch.float32,
    ):
        """
        Total and per-sample inference time on test set.

        Args:
            infer_dtype: Inference dtype (default float32). None uses cfg.dtype;
                torch.dtype or strings like "float16"/"bfloat16".

        Returns:
            dict: total_time_sec, n_samples, avg_time_per_sample_sec, infer_dtype
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

        test_loader_infer = DataLoader(
            self.test_set, batch_size=infer_batch_size, shuffle=False
        )

        def _sync():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        def _forward(epsilon_data, lambda_data, coord_data):
            eps = epsilon_data.to(self.device, dtype=infer_dtype, non_blocking=True)
            lam = lambda_data.to(self.device, dtype=infer_dtype, non_blocking=True)
            coord = coord_data.to(self.device, dtype=infer_dtype, non_blocking=True)
            return self.model(eps, lam, coord)

        with self._infer_precision(infer_dtype):
            for batch_idx, (_, epsilon_data, coord_data, _, _, lambda_data, *_) in enumerate(
                test_loader_infer
            ):
                _forward(epsilon_data, lambda_data, coord_data)
                if batch_idx + 1 >= warmup_batches:
                    break
            _sync()

            total_time = 0.0
            for _, epsilon_data, coord_data, _, _, lambda_data, *_ in test_loader_infer:
                _sync()
                t0 = time.perf_counter()
                _forward(epsilon_data, lambda_data, coord_data)
                _sync()
                total_time += time.perf_counter() - t0

        avg_time = total_time / n_test if n_test > 0 else 0.0
        print(f"Test samples: {n_test}")
        print(f"Inference dtype: {infer_dtype_tag(infer_dtype)}, training dtype: {infer_dtype_tag(orig_dtype)}")
        print(f"Total inference time: {total_time:.6f} s")
        print(f"Mean time per sample: {avg_time:.6f} s ({avg_time * 1000:.3f} ms)")

        self.model.train()
        return {
            "total_time_sec": total_time,
            "n_samples": n_test,
            "avg_time_per_sample_sec": avg_time,
            "infer_dtype": infer_dtype,
        }

if __name__ == "__main__":
    cfg = PINNConfig()
    # 2 branch (Epsilon + Lambda) + trunk
    # Improve SPP accuracy (same loss): multiscale Fourier trunk; try larger trunk_num_freq (6-8), hidden_channel, output_dim
    model = MIONet2Branch(branch_input_dim=2, trunk_input_dim=4, hidden_channel=256, output_dim=128, trunk_num_freq=4)
    model = model.double()
    pinn = PINN_maxwell(model, cfg)
    pinn.load_model()
    pinn.stat_test_inference_time(infer_batch_size=64,infer_dtype=torch.float16)
    #pinn.train(epochs=cfg.epochs, print_every=cfg.print_every, save_every=cfg.save_every)
    #pinn.plot_loss()
    #pinn.saveE_pred()



