import os
import datetime
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from getdata import GetDataset
from scipy.io import loadmat, savemat
from scipy.sparse.linalg import spilu
from torch.autograd import Function
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm


class ILUApply(Function):
    @staticmethod
    def forward(ctx, r_torch, ilu):
        ctx.ilu = ilu
        r_np = r_torch.detach().cpu().numpy()
        z_np = ilu.solve(r_np)
        z = torch.from_numpy(z_np).to(r_torch.device).to(r_torch.dtype)
        return z

    @staticmethod
    def backward(ctx, grad_out):
        ilu = ctx.ilu
        g_np = grad_out.detach().cpu().numpy()
        gr_np = ilu.solve(g_np, trans="H")
        grad_r = torch.from_numpy(gr_np).to(grad_out.device).to(grad_out.dtype)
        return grad_r, None


@dataclass
class PINNConfig:
    epochs: int = 1_000
    batch_size: int = 32
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    print_every: int = 1
    save_every: int = 1000

    matpath: str = "deepOnet_data_A1_1558"

    save_dir: str = "./model_save"
    results_dir: str = "./results"
    load_file_name: str = "model_A1_size_1558"
    save_file_name: str = "model_A1_size_1558"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float64
    num_workers: int = 10
    pin_memory: bool = True


class Modified_MLP_Block(nn.Module):
    def __init__(self, input_dim, hidden_channel, output_dim, hidden_size=6):
        super(Modified_MLP_Block, self).__init__()
        self.activation = nn.Tanh()
        self.encodeU = nn.Linear(input_dim, hidden_channel)
        self.encodeV = nn.Linear(input_dim, hidden_channel)
        self.In = nn.Linear(input_dim, hidden_channel)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_channel, hidden_channel) for _ in range(hidden_size)]
        )
        self.out = nn.Linear(hidden_channel, output_dim)
        self._init_weights()

    def _init_weights(self):
        torch.manual_seed(123)
        gain = nn.init.calculate_gain("tanh")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        u = self.activation(self.encodeU(x))
        v = self.activation(self.encodeV(x))
        hidden = self.activation(self.In(x))
        for layer in self.hidden_layers:
            z = self.activation(layer(hidden))
            hidden = (1 - z) * u + z * v
        x = self.out(hidden)
        return x


def _branch_norm2d(channels):
    return nn.InstanceNorm2d(channels)


def add_spatial_coord_channels(epsilon_data):
    bsz, _, h, w = epsilon_data.shape
    device, dtype = epsilon_data.device, epsilon_data.dtype
    j = (
        torch.linspace(0, 1, w, device=device, dtype=dtype)
        .view(1, 1, 1, w)
        .expand(bsz, 1, h, w)
    )
    i = (
        torch.linspace(0, 1, h, device=device, dtype=dtype)
        .view(1, 1, h, 1)
        .expand(bsz, 1, h, w)
    )
    return torch.cat([epsilon_data, j, i], dim=1)


class CNN_Branch_Residual(nn.Module):
    def __init__(self, in_channels=3, num_classes=128):
        super(CNN_Branch_Residual, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            _branch_norm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            _branch_norm2d(32),
            nn.ReLU(inplace=True),
        )
        self.res_block1 = ResidualBlock(32, 64, stride=2, norm_layer=_branch_norm2d)
        self.res_block2 = ResidualBlock(64, 128, stride=2, norm_layer=_branch_norm2d)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
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
    def __init__(self, in_channels, out_channels, stride=1, norm_layer=None):
        super(ResidualBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                norm_layer(out_channels),
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


class DeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_channel, output_dim):
        super(DeepONet, self).__init__()
        self.output_dim = output_dim
        self.branch_net = CNN_Branch_Residual(in_channels=3, num_classes=output_dim)
        self.trunk_net = Modified_MLP_Block(trunk_input_dim, hidden_channel, output_dim)

    def forward(self, branch_input, trunk_input):
        branch_input = add_spatial_coord_channels(branch_input)
        branch_out = self.branch_net(branch_input)
        trunk_out = self.trunk_net(trunk_input)
        b1, b2 = (
            branch_out[:, : self.output_dim // 2],
            branch_out[:, self.output_dim // 2 :],
        )
        t1, t2 = (
            trunk_out[:, :, : self.output_dim // 2],
            trunk_out[:, :, self.output_dim // 2 :],
        )
        s_re = torch.einsum("bi,bni->bn", b1, t1)
        s_im = torch.einsum("bi,bni->bn", b2, t2)
        return s_re, s_im


class PINN_maxwell:
    def __init__(self, model, config: PINNConfig):
        self.cfg = config
        self.ddp = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.ddp else 0
        self.world_size = dist.get_world_size() if self.ddp else 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device, dtype=self.cfg.dtype)
        if self.ddp:
            if self.device.type == "cuda":
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                )
            else:
                self.model = DDP(self.model)

        self.batch_size = self.cfg.batch_size
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.epochs,
            eta_min=self.cfg.min_learning_rate,
        )

        self.losses = []
        self.save_file_name = self.cfg.save_file_name
        self.load_file_name = self.cfg.load_file_name
        self.save_dir = Path(self.cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.train_set, self.test_set = self.load_dataset()
        self.train_sampler = (
            DistributedSampler(
                self.train_set,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            if self.ddp
            else None
        )
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
        )
        self.test_loader = DataLoader(
            self.test_set, batch_size=len(self.test_set), shuffle=False
        )
        self.ilu_cache = {}
        self._build_ilu_cache()

    def _model_state_dict(self):
        if isinstance(self.model, DDP):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def _reduce_mean(self, value):
        value_tensor = torch.tensor(value, device=self.device, dtype=torch.float64)
        if self.ddp:
            dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
            value_tensor /= self.world_size
        return value_tensor.item()

    def load_model(self):
        state = torch.load(
            self.save_dir / f"{self.load_file_name}.pth",
            map_location=self.device,
            weights_only=True,
        )
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(state)
        else:
            self.model.load_state_dict(state)

    def E_function(self, epsilon_data, coord_data):
        epsilon_data = epsilon_data.to(self.device)
        coord_data = coord_data.to(self.device)
        return self.model(epsilon_data, coord_data)

    def get_fem_loss(
        self, indices, epsilon_data, coord_data, e_true, ai, aj, av, b, coord_len
    ):
        ere_pred, eim_pred = self.E_function(epsilon_data, coord_data)
        e = torch.complex(ere_pred, eim_pred)
        bsz, mmax = e.shape
        mi = coord_len.squeeze(-1).long().to(self.device)
        arange_m = torch.arange(mmax, device=self.device)
        mask_x = arange_m[None, :] < mi[:, None]
        x_flat = e[mask_x]
        b_flat = b.to(self.device)[mask_x].to(x_flat.dtype)
        sum_mi = int(mi.sum().item())
        offsets = torch.cumsum(
            torch.cat([torch.zeros(1, device=self.device, dtype=torch.long), mi[:-1]]),
            dim=0,
        )

        ai = ai.to(self.device).long()
        aj = aj.to(self.device).long()
        av = av.to(self.device).to(x_flat.dtype)
        mask_nnz = (ai > 0) & (aj > 0)
        rows = (ai - 1 + offsets.unsqueeze(1)).masked_select(mask_nnz)
        cols = (aj - 1 + offsets.unsqueeze(1)).masked_select(mask_nnz)
        vals = av.masked_select(mask_nnz)

        y = torch.zeros(sum_mi, dtype=x_flat.dtype, device=self.device)
        y.scatter_add_(0, rows, vals * x_flat.index_select(0, cols))
        r = y - b_flat

        z_parts = []
        for i in range(bsz):
            start = int(offsets[i].item())
            m = int(mi[i].item())
            r_i = r[start : start + m]
            ilu = self.ilu_cache[int(indices[i].item())]
            z_i = ILUApply.apply(r_i, ilu)
            z_parts.append(z_i)
        z = torch.cat(z_parts, dim=0)
        loss = (z.abs() ** 2).mean()
        return loss

    @torch.no_grad()
    def test_E_loss(self):
        if self.ddp and self.rank != 0:
            return 0.0
        self.model.eval()
        total_loss = 0.0
        for (
            indices,
            epsilon_data,
            coord_data,
            e_true,
            ai,
            aj,
            av,
            b,
            coord_len,
        ) in self.test_loader:
            indices = indices.to(self.device)
            epsilon_data = epsilon_data.to(self.device)
            coord_data = coord_data.to(self.device)
            loss = self.get_fem_loss(
                indices, epsilon_data, coord_data, e_true, ai, aj, av, b, coord_len
            )
            total_loss += loss.item()
        self.model.train()
        return total_loss / len(self.test_loader)

    def train(self, epochs, print_every=100, save_every=10000):
        if self.rank == 0:
            self.losses.append(["epoch", "fem_loss", "test_loss"])
        start_time = datetime.datetime.now()
        epoch_iter = tqdm(
            range(epochs), desc=f"Training rank{self.rank}", disable=(self.rank != 0)
        )
        for epoch in epoch_iter:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self.model.train()
            total_loss = 0.0
            for (
                indices,
                epsilon_data,
                coord_data,
                e_true,
                ai,
                aj,
                av,
                b,
                coord_len,
            ) in self.train_loader:
                indices = indices.to(self.device)
                epsilon_data = epsilon_data.to(self.device)
                coord_data = coord_data.to(self.device)
                self.optimizer.zero_grad()
                fem_loss = self.get_fem_loss(
                    indices, epsilon_data, coord_data, e_true, ai, aj, av, b, coord_len
                )
                loss = fem_loss
                loss.backward()
                if self.cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.max_grad_norm
                    )
                self.optimizer.step()
                total_loss += loss.item()

            local_avg = total_loss / max(1, len(self.train_loader))
            global_avg = self._reduce_mean(local_avg)
            avg_test_loss = self.test_E_loss()
            self.scheduler.step()

            if self.rank == 0:
                self.losses.append([epoch, global_avg, avg_test_loss])
                if epoch % print_every == 0:
                    print(
                        f"Epoch {epoch}, Total Loss: {global_avg}, test Loss {avg_test_loss}"
                    )
                if (epoch + 1) % save_every == 0:
                    ckpt_path = (
                        self.save_dir / f"{self.save_file_name}_epoch{epoch + 1}.pth"
                    )
                    torch.save(self._model_state_dict(), ckpt_path)

        if self.rank == 0:
            torch.save(self._model_state_dict(), self.save_file_name)
            print("Current learning rate:", self.optimizer.param_groups[0]["lr"])
            print(
                "Training Time:",
                (datetime.datetime.now() - start_time).total_seconds(),
                "s",
            )

    def plot_loss(self):
        if self.rank != 0 or len(self.losses) <= 1:
            return
        data = np.array(self.losses[1:])
        epochs = data[:, 0]
        train_loss = data[:, 1]
        test_loss = data[:, 2]
        plt.figure(figsize=(10, 6))
        plt.title("Training/Test Loss")
        plt.semilogy(epochs, train_loss, label="train_loss")
        plt.semilogy(epochs, test_loss, label="test_loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        save_path = "results/loss_plot2_1_ddp.png"
        plt.savefig(save_path)
        plt.show()

    def load_dataset(self):
        data_set = loadmat(self.cfg.matpath)
        epsilon_train = data_set["Eplison_train"]
        x_train = data_set["X_train"]
        ez_train = data_set["Ez_train"]
        epsilon_test = data_set["Eplison_test"]
        x_test = data_set["X_test"]
        ez_test = data_set["Ez_test"]
        coord_len_train = data_set["coord_len_train"]
        coord_len_test = data_set["coord_len_test"]
        ai_train, aj_train = data_set["Ai_train"], data_set["Aj_train"]
        ai_test, aj_test = data_set["Ai_test"], data_set["Aj_test"]
        av_train, av_test = data_set["Av_train"], data_set["Av_test"]
        b_train, b_test = data_set["b_train"], data_set["b_test"]

        n_train, n_test = len(epsilon_train), len(epsilon_test)
        if self.rank == 0:
            print(f"Train samples: {n_train}, Test samples: {n_test}")
            print(
                f"Train shapes: ε {epsilon_train.shape}, X {x_train.shape}, Ez {ez_train.shape}"
            )

        train_dataset = GetDataset(
            epsilon_train,
            x_train,
            ez_train,
            ai_train,
            aj_train,
            av_train,
            b_train,
            coord_len_train,
            index_offset=0,
        )
        test_dataset = GetDataset(
            epsilon_test,
            x_test,
            ez_test,
            ai_test,
            aj_test,
            av_test,
            b_test,
            coord_len_test,
            index_offset=n_train,
        )
        return train_dataset, test_dataset

    def _build_ilu_cache(self):
        train_iter = tqdm(
            range(len(self.train_set)),
            desc=f"Building ILU cache (train) rank{self.rank}",
            disable=(self.rank != 0),
        )
        for idx in train_iter:
            ai = self.train_set.Ai[idx].numpy()
            aj = self.train_set.Aj[idx].numpy()
            av = self.train_set.Av[idx].numpy()
            mi = int(self.train_set.coord_len[idx].item())
            mask = (ai > 0) & (aj > 0)
            rows = (ai[mask] - 1).astype(np.int64)
            cols = (aj[mask] - 1).astype(np.int64)
            vals = av[mask]
            a = sp.coo_matrix((vals, (rows, cols)), shape=(mi, mi)).tocsc()
            self.ilu_cache[idx] = spilu(a)

        test_iter = tqdm(
            range(len(self.test_set)),
            desc=f"Building ILU cache (test) rank{self.rank}",
            disable=(self.rank != 0),
        )
        for idx in test_iter:
            ai = self.test_set.Ai[idx].numpy()
            aj = self.test_set.Aj[idx].numpy()
            av = self.test_set.Av[idx].numpy()
            mi = int(self.test_set.coord_len[idx].item())
            mask = (ai > 0) & (aj > 0)
            rows = (ai[mask] - 1).astype(np.int64)
            cols = (aj[mask] - 1).astype(np.int64)
            vals = av[mask]
            a = sp.coo_matrix((vals, (rows, cols)), shape=(mi, mi)).tocsc()
            self.ilu_cache[len(self.train_set) + idx] = spilu(a)

        if self.rank == 0:
            print(f"ILU cache built: {len(self.ilu_cache)} samples.")

    @torch.no_grad()
    def saveE_pred(self):
        if self.rank != 0:
            return
        self.model.eval()
        infer_batch_size = 64
        n_train = len(self.train_set)
        n_test = len(self.test_set)
        mmax_train = self.train_set.coord.shape[1]
        mmax_test = self.test_set.coord.shape[1]
        e_train_pred = np.zeros((n_train, mmax_train), dtype=np.complex128)
        e_test_pred = np.zeros((n_test, mmax_test), dtype=np.complex128)

        train_loader_infer = DataLoader(
            self.train_set, batch_size=infer_batch_size, shuffle=False
        )
        for indices, epsilon_data, coord_data, e_true, *_ in train_loader_infer:
            idx = indices.cpu().numpy()
            epsilon_data = epsilon_data.to(self.device)
            coord_data = coord_data.to(self.device)
            e_real, e_imag = self.E_function(epsilon_data, coord_data)
            e_pred = torch.complex(e_real, e_imag).detach().cpu().numpy()
            m_batch = e_pred.shape[1]
            e_train_pred[idx, :m_batch] = e_pred

        test_loader_infer = DataLoader(
            self.test_set, batch_size=infer_batch_size, shuffle=False
        )
        for indices, epsilon_data, coord_data, e_true, *_ in test_loader_infer:
            idx_global = indices.cpu().numpy()
            idx_local = idx_global - n_train
            epsilon_data = epsilon_data.to(self.device)
            coord_data = coord_data.to(self.device)
            e_real, e_imag = self.E_function(epsilon_data, coord_data)
            e_pred = torch.complex(e_real, e_imag).detach().cpu().numpy()
            m_batch = e_pred.shape[1]
            e_test_pred[idx_local, :m_batch] = e_pred

        savemat("E_train_pred_size_1558.mat", {"E_pred": e_train_pred})
        savemat("E_test_pred_size_1558.mat", {"E_pred": e_test_pred})
        self.model.train()


def setup_distributed():
    use_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not use_ddp:
        return False
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    return True


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main():
    use_ddp = setup_distributed()
    cfg = PINNConfig()
    model = DeepONet(
        branch_input_dim=1, trunk_input_dim=2, hidden_channel=128, output_dim=128
    )
    model = model.double()
    pinn = PINN_maxwell(model, cfg)
    pinn.train(
        epochs=cfg.epochs, print_every=cfg.print_every, save_every=cfg.save_every
    )
    pinn.plot_loss()
    pinn.saveE_pred()
    if use_ddp:
        cleanup_distributed()


if __name__ == "__main__":
    main()
