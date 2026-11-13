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
import pandas as pd

# Plot modules
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path
from scipy.io import savemat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PINNConfig:

    epochs: int = 1_000
    batch_size: int = 32
    learning_rate: float = 1e-2
    step_size: int = 500
    gamma: float = 0.95
    max_grad_norm: float = 1.0
    print_every: int = 1
    save_every: int = 1000

    matpath: str = "deepOnet_data_C1_3388.mat"

    save_dir: str = "./model_save"
    results_dir: str = "./results"
    load_file_name: str = "model_A1_size_847"
    save_file_name: str = "model_A1_size_847"

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
        U = self.activation(self.encodeU(x))
        V = self.activation(self.encodeV(x))
        Hidden = self.activation(self.In(x))

        for layer in self.hidden_layers:
            Z = self.activation(layer(Hidden))
            Hidden = (1 - Z) * U + Z * V

        x = self.out(Hidden)
        return x


def _branch_norm2d(channels):
    """Branch norm2d."""
    return nn.InstanceNorm2d(channels)


def add_spatial_coord_channels(epsilon_data):
    """Add spatial coord channels."""
    B, _, H, W = epsilon_data.shape
    device, dtype = epsilon_data.device, epsilon_data.dtype
    j = (
        torch.linspace(0, 1, W, device=device, dtype=dtype)
        .view(1, 1, 1, W)
        .expand(B, 1, H, W)
    )
    i = (
        torch.linspace(0, 1, H, device=device, dtype=dtype)
        .view(1, 1, H, 1)
        .expand(B, 1, H, W)
    )
    return torch.cat([epsilon_data, j, i], dim=1)


class CNN_Branch_Residual(nn.Module):
    """Cnn branch residual."""

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
    """Residualblock."""

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

        E_pred = torch.einsum("bi,bni->bn", branch_out, trunk_out)
        return E_pred


class PINN_maxwell:
    def __init__(self, model, config: PINNConfig):
        self.cfg = config

        self.device = torch.device(self.cfg.device)
        self.model = model.to(self.device, dtype=self.cfg.dtype)
        self.batch_size = self.cfg.batch_size
        self.learning_rate = self.cfg.learning_rate
        self.matpath = self.cfg.matpath
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.cfg.step_size, gamma=self.cfg.gamma
        )
        self.losses = []
        self.lamda = []
        self.save_file_name = self.cfg.save_file_name
        self.load_file_name = self.cfg.load_file_name
        self.save_dir = Path(self.cfg.save_dir)
        self.train_set, self.test_set = self.load_dataset()
        self.train_loader = DataLoader(
            self.train_set, self.cfg.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_set, batch_size=len(self.test_set), shuffle=False
        )

    def load_model(self):
        self.model.load_state_dict(
            torch.load(
                self.save_dir / f"{self.load_file_name}.pth",
                map_location=self.device,
                weights_only=True,
            )
        )

    def E_function(self, epsilon_data, coord_data):
        epsilon_data = epsilon_data.to(self.device)
        coord_data = coord_data.to(self.device)
        return self.model(epsilon_data, coord_data)

    def get_data_loss(self, epsilon_data, coord_data, E_true):
        E_pred = self.E_function(epsilon_data, coord_data)

        E_true_real = E_true[:, :, 0] if E_true.dim() == 3 else E_true
        data_loss = self.loss_fn(E_pred, E_true_real)
        return data_loss

    def get_fem_loss(
        self, indices, epsilon_data, coord_data, E_true, Ai, Aj, Av, b, coord_len
    ):
        """Get fem loss."""
        E = self.E_function(epsilon_data, coord_data)

        B, Mmax = E.shape
        Mi = coord_len.squeeze(-1).long().to(self.device)  # shape: (B,)

        arangeM = torch.arange(Mmax, device=self.device)  # (Mmax,)
        mask_x = arangeM[None, :] < Mi[:, None]  # (B, Mmax)

        x_flat = E[mask_x]  # (sum Mi,)
        b_flat = b.to(self.device)[mask_x].to(x_flat.dtype)  # (sum Mi,)

        sumMi = int(Mi.sum().item())
        offsets = torch.cumsum(
            torch.cat([torch.zeros(1, device=self.device, dtype=torch.long), Mi[:-1]]),
            dim=0,
        )  # (B,)

        Ai = Ai.to(self.device).long()
        Aj = Aj.to(self.device).long()
        Av = Av.to(self.device).to(x_flat.dtype)
        mask_nnz = (Ai > 0) & (Aj > 0)

        rows = (Ai - 1 + offsets.unsqueeze(1)).masked_select(mask_nnz)
        cols = (Aj - 1 + offsets.unsqueeze(1)).masked_select(mask_nnz)
        vals = Av.masked_select(mask_nnz)

        y = torch.zeros(sumMi, dtype=x_flat.dtype, device=self.device)
        y.scatter_add_(0, rows, vals * x_flat.index_select(0, cols))
        r = y - b_flat
        xTAx = torch.sum(x_flat * y)
        xTb = torch.sum(x_flat * b_flat)
        energy = 0.5 * xTAx - xTb
        loss = energy
        return loss

    @torch.no_grad()
    def test_E_loss(self):
        self.model.eval()
        total_loss = 0.0
        for (
            indices,
            epsilon_data,
            coord_data,
            E_true,
            Ai,
            Aj,
            Av,
            b,
            coord_len,
        ) in self.test_loader:
            indices = indices.to(self.device)
            epsilon_data = epsilon_data.to(self.device)
            coord_data = coord_data.to(self.device)
            loss = self.get_fem_loss(
                indices, epsilon_data, coord_data, E_true, Ai, Aj, Av, b, coord_len
            )
            total_loss += loss.item()
        self.model.train()
        return total_loss / len(self.test_loader)

    def train(self, epochs, print_every=100, save_every=10000):
        self.losses.append(["epoch", "fem_loss", "test_loss"])
        start_time = datetime.datetime.now()
        for epoch in tqdm(range(epochs), desc="Training"):
            self.model.train()
            total_loss = 0.0
            fem_loss = 0.0

            for (
                indices,
                epsilon_data,
                coord_data,
                E_true,
                Ai,
                Aj,
                Av,
                b,
                coord_len,
            ) in self.train_loader:
                indices = indices.to(self.device)
                epsilon_data = epsilon_data.to(self.device)
                coord_data = coord_data.to(self.device)
                self.optimizer.zero_grad()

                fem_loss = self.get_fem_loss(
                    indices, epsilon_data, coord_data, E_true, Ai, Aj, Av, b, coord_len
                )

                loss = fem_loss

                loss.backward()
                if self.cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.max_grad_norm
                    )
                self.optimizer.step()

                total_loss += loss.item()

            avg_total_loss = total_loss / len(self.train_loader)
            avg_test_loss = self.test_E_loss()
            self.losses.append([epoch, avg_total_loss, avg_test_loss])
            self.scheduler.step()

            if epoch % print_every == 0:
                print(
                    f"Epoch {epoch}, Total Loss: {avg_total_loss}, test Loss {avg_test_loss}"
                )
            if (epoch + 1) % save_every == 0:
                ckpt_path = (
                    self.save_dir / f"{self.save_file_name}_epoch{epoch + 1}.pth"
                )
                torch.save(self.model.state_dict(), ckpt_path)
        torch.save(self.model.state_dict(), self.save_file_name)
        print("Current learning rate:", self.optimizer.param_groups[0]["lr"])
        print(
            "Training Time:",
            (datetime.datetime.now() - start_time).total_seconds(),
            "s",
        )

    def plot_loss(self):
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
        save_path = "results/loss_plot2_1.png"
        plt.savefig(save_path)
        plt.show()

    def load_dataset(self):
        """Load dataset."""
        data_set = loadmat(self.matpath)

        Epsilon_train = data_set["Epsilon_train"]
        X_train = data_set["X_train"]
        Ez_train = data_set["Ez_train"]

        Epsilon_test = data_set["Epsilon_test"]
        X_test = data_set["X_test"]
        Ez_test = data_set["Ez_test"]

        coord_len_train = data_set["coord_len_train"]
        coord_len_test = data_set["coord_len_test"]

        Ai_train, Aj_train = data_set["Ai_train"], data_set["Aj_train"]
        Ai_test, Aj_test = data_set["Ai_test"], data_set["Aj_test"]
        Av_train, Av_test = data_set["Av_train"], data_set["Av_test"]
        b_train, b_test = data_set["b_train"], data_set["b_test"]

        n_train, n_test = len(Epsilon_train), len(Epsilon_test)
        print(f"Train samples: {n_train}, Test samples: {n_test}")
        print(
            f"Train shapes: ε {Epsilon_train.shape}, X {X_train.shape}, Ez {Ez_train.shape}"
        )

        Train_dataset = GetDataset(
            Epsilon_train,
            X_train,
            Ez_train,
            Ai_train,
            Aj_train,
            Av_train,
            b_train,
            coord_len_train,
            index_offset=0,
        )
        Test_dataset = GetDataset(
            Epsilon_test,
            X_test,
            Ez_test,
            Ai_test,
            Aj_test,
            Av_test,
            b_test,
            coord_len_test,
            index_offset=n_train,
        )
        return Train_dataset, Test_dataset

    @torch.no_grad()
    def saveE_pred(self):
        """Savee pred."""
        self.model.eval()
        infer_batch_size = 64
        n_train = len(self.train_set)
        n_test = len(self.test_set)

        Mmax_train = self.train_set.coord.shape[1]
        Mmax_test = self.test_set.coord.shape[1]

        E_train_pred = np.zeros((n_train, Mmax_train), dtype=np.float64)
        E_test_pred = np.zeros((n_test, Mmax_test), dtype=np.float64)

        for indices, epsilon_data, coord_data, *_ in self.train_loader:
            idx = indices.cpu().numpy()
            epsilon_data = epsilon_data.to(self.device)
            coord_data = coord_data.to(self.device)
            E_pred = self.E_function(epsilon_data, coord_data).detach().cpu().numpy()
            M_batch = E_pred.shape[1]
            E_train_pred[idx, :M_batch] = E_pred

        test_loader_infer = DataLoader(
            self.test_set, batch_size=infer_batch_size, shuffle=False
        )
        for indices, epsilon_data, coord_data, *_ in test_loader_infer:
            idx_global = indices.cpu().numpy()
            idx_local = idx_global - n_train
            epsilon_data = epsilon_data.to(self.device)
            coord_data = coord_data.to(self.device)
            E_pred = self.E_function(epsilon_data, coord_data).detach().cpu().numpy()
            M_batch = E_pred.shape[1]
            E_test_pred[idx_local, :M_batch] = E_pred

        savemat("E_train_pred_size_847.mat", {"E_pred": E_train_pred})
        savemat("E_test_pred_size_847.mat", {"E_pred": E_test_pred})
        self.model.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(Path(__file__).resolve().parent / "deepOnet_data_C1_3388.mat"),
    )
    args = parser.parse_args()
    cfg = PINNConfig(matpath=args.data)
    model = DeepONet(
        branch_input_dim=1, trunk_input_dim=2, hidden_channel=128, output_dim=64
    )
    model = model.double()
    pinn = PINN_maxwell(model, cfg)
    # pinn.load_model()
    pinn.train(
        epochs=cfg.epochs, print_every=cfg.print_every, save_every=cfg.save_every
    )
    # pinn.plot_loss()
    pinn.saveE_pred()
