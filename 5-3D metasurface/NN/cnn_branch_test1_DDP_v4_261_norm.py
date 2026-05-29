import datetime
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import cnn_branch_test1_DDP as base


@dataclass
class PINNConfigV4Norm(base.PINNConfig):
    """
    V4-Norm goals:
    1) Lambda branch: direct MLP input (no FFT);
    2) Trunk: FFM encoding;
    3) Loss logging during training;
    4) Lambda normalized in branch to given physical range.
    """
    batch_size: int = 1
    learning_rate: float = 1e-3
    dtype: torch.dtype = torch.float32
    save_file_name: str = "model_3Dcase3_size_261_ddp_v4_norm"
    load_file_name: str = "model_3Dcase3_size_261_ddp_v4_norm_epoch1000"
    lambda_norm_min: float = 0.54
    lambda_norm_max: float = 0.8


class LambdaDirectMLPWithNorm(nn.Module):
    """Normalize lambda to interval, then MLP (no FFT / spectral transform)."""

    def __init__(
        self,
        lambda_input_dim: int,
        output_dim_2bi: int,
        hidden_mult: int = 4,
        num_layers: int = 3,
        lambda_min: float = 0.54,
        lambda_max: float = 0.8,
        eps: float = 1e-12,
    ):
        super().__init__()
        in_dim = lambda_input_dim
        bi = max(output_dim_2bi // 2, 1)
        hidden = max(output_dim_2bi, hidden_mult * bi)

        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.eps = eps

        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, output_dim_2bi)]
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def _normalize_lambda(self, lambda_in: torch.Tensor) -> torch.Tensor:
        # Min-max to [0,1] over known physical range, then map to [-1,1] and clamp.
        # Better match for Tanh; more stable near boundaries / rapid variation.
        denom = self.lambda_max - self.lambda_min
        lambda_norm01 = (lambda_in - self.lambda_min) / denom
        lambda_norm01 = torch.clamp(lambda_norm01, 0.0, 1.0)
        lambda_scaled = lambda_norm01 * 2.0 - 1.0
        return lambda_scaled

    def forward(self, lambda_in: torch.Tensor) -> torch.Tensor:
        lambda_norm = self._normalize_lambda(lambda_in)
        return self.mlp(lambda_norm)


class FFMEncoding(nn.Module):
    """
    FFM / Random Fourier Features:
    phi(x) = [sin(2*pi*xB), cos(2*pi*xB)].
    """

    def __init__(self, in_dim: int, num_features: int = 64, sigma: float = 1.58):
        super().__init__()
        B = torch.randn(in_dim, num_features) * sigma
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * np.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class FFM_MLP_Trunk(nn.Module):
    """Trunk: FFM encoding + Modified_MLP_Block (no FFT)."""

    def __init__(
        self,
        trunk_input_dim: int,
        hidden_channel: int,
        output_dim: int,
        hidden_size: int = 8,
        ffm_features: int = 32,
        ffm_sigma: float = 1.58,
    ):
        super().__init__()
        self.ffm = FFMEncoding(trunk_input_dim, num_features=ffm_features, sigma=ffm_sigma)
        self.mlp = base.Modified_MLP_Block(
            input_dim=2 * ffm_features,
            hidden_channel=hidden_channel,
            output_dim=output_dim,
            hidden_size=hidden_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ffm = self.ffm(x)
        return self.mlp(x_ffm)


class MIONetV4Norm(nn.Module):
    """V4-Norm: epsilon branch unchanged; lambda branch direct MLP+norm; trunk uses FFM."""

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        hidden_channel: int,
        output_dim: int,
        ffm_features: int = 64,
        ffm_sigma: float = 1.58,
        lambda_norm_min: float = 0.54,
        lambda_norm_max: float = 0.8,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.bi = output_dim // 2
        self.bni = output_dim // 2

        self.branch_net = base.CNN_Branch_Residual(in_channels=branch_input_dim + 3, num_classes=output_dim)
        self.branch_net_lambda = LambdaDirectMLPWithNorm(
            lambda_input_dim=1,
            output_dim_2bi=2 * self.bi,
            lambda_min=lambda_norm_min,
            lambda_max=lambda_norm_max,
        )
        self.trunk_net = FFM_MLP_Trunk(
            trunk_input_dim=trunk_input_dim,
            hidden_channel=hidden_channel,
            output_dim=output_dim,
            hidden_size=8,
            ffm_features=ffm_features,
            ffm_sigma=ffm_sigma,
        )

    def forward(self, branch_input_epsilon, branch_input_lambda, trunk_input):
        branch_input_epsilon = base.add_spatial_coord_channels(branch_input_epsilon)
        branch_out_eps = self.branch_net(branch_input_epsilon)
        B11 = branch_out_eps[:, :self.bi]
        B21 = branch_out_eps[:, self.bi:]

        branch_out_lambda = self.branch_net_lambda(branch_input_lambda)
        B31 = branch_out_lambda[:, :self.bi]
        B32 = branch_out_lambda[:, self.bi:]

        trunk_out = self.trunk_net(trunk_input)
        T1 = trunk_out[:, :, :self.bni]
        T2 = trunk_out[:, :, self.bni:]

        s_re = torch.einsum("bi,bni->bn", B11 * B31, T1)
        s_im = torch.einsum("bi,bni->bn", B21 * B32, T2)
        return s_re, s_im


class PINNMaxwellV4Norm(base.PINN_maxwell):
    """
    Enhancements:
    - Unified dtype (no float64/float32 mix);
    - Per-epoch loss logs (txt + csv).
    """

    def E_function(self, epsilon_data, lambda_data, coord_data):
        epsilon_data = epsilon_data.to(self.device, dtype=self.cfg.dtype)
        lambda_data = lambda_data.to(self.device, dtype=self.cfg.dtype)
        coord_data = coord_data.to(self.device, dtype=self.cfg.dtype)
        return self.model(epsilon_data, lambda_data, coord_data)

    def train(self, epochs, print_every=100, save_every=10000):
        self.losses.append(["epoch", "fem_loss", "test_loss"])
        best_test_loss = float("inf")
        start_time = datetime.datetime.now()

        log_dir = Path(self.cfg.results_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_log_path = log_dir / f"{self.save_file_name}_loss_log.csv"
        txt_log_path = log_dir / f"{self.save_file_name}_loss_log.txt"

        with open(csv_log_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["time", "epoch", "epochs", "train_fem", "test_fem", "lr", "epoch_time_s"])

        with open(txt_log_path, "w", encoding="utf-8") as f_txt:
            f_txt.write("epoch-wise loss log\n")

        for epoch in base.tqdm(range(epochs), desc="Training", disable=not self.is_main):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            self.model.train()
            total_loss = 0.0
            epoch_start = datetime.datetime.now()

            for indices, epsilon_data, coord_data, E_true, lambda_data, Ai, Aj, Av, b, coord_len in self.train_loader:
                indices = indices.to(self.device)
                epsilon_data = epsilon_data.to(self.device, dtype=self.cfg.dtype)
                coord_data = coord_data.to(self.device, dtype=self.cfg.dtype)
                lambda_data = lambda_data.to(self.device, dtype=self.cfg.dtype)
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

            if self.is_main:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    best_ckpt_path = self.save_dir / f"{self.save_file_name}_best.pth"
                    torch.save(self._raw_model().state_dict(), best_ckpt_path)

                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                lr = self.optimizer.param_groups[0]["lr"]
                epoch_seconds = (datetime.datetime.now() - epoch_start).total_seconds()
                log_line = (
                    f"[{now_str}] Epoch {epoch}/{epochs - 1} | "
                    f"train_fem={avg_total_loss:.6e} | "
                    f"test_fem={avg_test_loss:.6e} | "
                    f"lr={lr:.3e} | "
                    f"epoch_time={epoch_seconds:.2f}s"
                )

                if epoch % print_every == 0:
                    print(log_line)

                with open(txt_log_path, "a", encoding="utf-8") as f_txt:
                    f_txt.write(log_line + "\n")
                with open(csv_log_path, "a", newline="", encoding="utf-8") as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerow([now_str, epoch, epochs - 1, avg_total_loss, avg_test_loss, lr, epoch_seconds])

                if (epoch + 1) % save_every == 0:
                    ckpt_path = self.save_dir / f"{self.save_file_name}_epoch{epoch + 1}.pth"
                    torch.save(self._raw_model().state_dict(), ckpt_path)

        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self._raw_model().state_dict(), self.save_dir / f"{self.save_file_name}.pth")
            total_seconds = (datetime.datetime.now() - start_time).total_seconds()
            print("Current learning rate:", self.optimizer.param_groups[0]["lr"])
            print("Training Time:", total_seconds, "s")
            with open(txt_log_path, "a", encoding="utf-8") as f_txt:
                f_txt.write(f"Current learning rate: {self.optimizer.param_groups[0]['lr']}\n")
                f_txt.write(f"Training Time: {total_seconds} s\n")


if __name__ == "__main__":
    rank, world_size, local_rank, device = base.setup_ddp()
    cfg = PINNConfigV4Norm(build_ilu_cache=False)

    model = MIONetV4Norm(
        branch_input_dim=2,
        trunk_input_dim=6,
        hidden_channel=512,
        output_dim=256,
        ffm_features=64,
        ffm_sigma=1.58,
        lambda_norm_min=cfg.lambda_norm_min,
        lambda_norm_max=cfg.lambda_norm_max,
    ).to(dtype=cfg.dtype)
    pinn = PINNMaxwellV4Norm(model, cfg, rank=rank, world_size=world_size, local_rank=local_rank, device=device)
    pinn.load_model()
    pinn.stat_test_inference_time(infer_dtype=torch.float16)
    #pinn.train(epochs=cfg.epochs, print_every=cfg.print_every, save_every=cfg.save_every)
    #pinn.plot_loss()
    #pinn.saveE_pred()
    base.cleanup_ddp()
