import torch
import torch.nn as nn

from cnn_branch_test1_DDP import (
    PINNConfig,
    PINN_maxwell,
    setup_ddp,
    cleanup_ddp,
    CNN_Branch_Residual,
    Modified_MLP_Block,
    FourierFeatureMapping,
)


class FFT_MLP_Branch(nn.Module):
    """Lambda branch: FFT then MLP."""

    def __init__(self, lambda_input_dim, output_dim_2bi, hidden_mult=4, num_layers=3):
        super(FFT_MLP_Branch, self).__init__()
        in_dim = 2 * lambda_input_dim
        bi = output_dim_2bi // 2
        hidden = max(output_dim_2bi, hidden_mult * max(bi, 1))

        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
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
        x_fft = torch.fft.fft(lambda_in, dim=-1)
        x_cat = torch.cat([x_fft.real, x_fft.imag], dim=-1)
        return self.mlp(x_cat)


class MIONet(nn.Module):
    """MIONet: 3 branches + FFM trunk; lambda branch uses FFT."""

    def __init__(self, branch_input_dim, trunk_input_dim, hidden_channel, output_dim, ffm_mapping_size=64, ffm_scale=2.0):
        super(MIONet, self).__init__()
        self.output_dim = output_dim
        self.bi = output_dim // 2
        self.bni = output_dim // 2

        self.branch_net = CNN_Branch_Residual(in_channels=branch_input_dim + 2, num_classes=output_dim)
        self.branch_net_ebz = CNN_Branch_Residual(in_channels=branch_input_dim + 2, num_classes=output_dim)
        self.branch_net_lambda = FFT_MLP_Branch(lambda_input_dim=1, output_dim_2bi=2 * self.bi)

        self.ffm = FourierFeatureMapping(input_dim=trunk_input_dim, mapping_size=ffm_mapping_size, scale=ffm_scale)
        ffm_output_dim = ffm_mapping_size * 2
        self.trunk_net = Modified_MLP_Block(ffm_output_dim, hidden_channel, output_dim)

    def forward(self, branch_input_epsilon, branch_input_ebz, branch_input_lambda, trunk_input):
        from cnn_branch_test1_DDP import add_spatial_coord_channels

        branch_input_epsilon = add_spatial_coord_channels(branch_input_epsilon)
        branch_out_eps = self.branch_net(branch_input_epsilon)
        B11 = branch_out_eps[:, :self.bi]
        B21 = branch_out_eps[:, self.bi:]

        branch_input_ebz = add_spatial_coord_channels(branch_input_ebz)
        branch_out_ebz = self.branch_net_ebz(branch_input_ebz)
        B12 = branch_out_ebz[:, :self.bi]
        B22 = branch_out_ebz[:, self.bi:]

        branch_out_lambda = self.branch_net_lambda(branch_input_lambda)
        B31 = branch_out_lambda[:, :self.bi]
        B32 = branch_out_lambda[:, self.bi:]

        trunk_input_ffm = self.ffm(trunk_input)
        trunk_out = self.trunk_net(trunk_input_ffm)
        T1 = trunk_out[:, :, :self.bni]
        T2 = trunk_out[:, :, self.bni:]

        s_re = torch.einsum("bi,bni->bn", B11 * B12 * B31, T1)
        s_im = torch.einsum("bi,bni->bn", B21 * B22 * B32, T2)
        return s_re, s_im

if __name__ == "__main__":

    rank, world_size, local_rank, device = setup_ddp()

    cfg = PINNConfig()

    cfg.load_file_name = f"{cfg.load_file_name}"

    cfg.save_file_name = f"{cfg.save_file_name}"



    model = MIONet(branch_input_dim=2, trunk_input_dim=2, hidden_channel=512, output_dim=256, ffm_scale=1.58)

    model = model.double()



    pinn = PINN_maxwell(model, cfg, rank=rank, world_size=world_size, local_rank=local_rank, device=device)
    
    pinn.load_model()

    pinn.stat_test_inference_time(infer_dtype=torch.float32)

    # pinn.train(epochs=cfg.epochs, print_every=cfg.print_every, save_every=cfg.save_every)

    #pinn.plot_loss()

    # pinn.saveE_pred()

    cleanup_ddp()