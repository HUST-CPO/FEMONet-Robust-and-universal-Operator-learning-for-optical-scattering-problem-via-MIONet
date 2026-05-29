from scipy.interpolate import griddata
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from scipy.io import loadmat

DATA_DIR = Path(__file__).resolve().parent.parent / "samples and post proceeding" / "mat_data"
class GetDataset(Dataset):
    def __init__(self, epsilon=None, coord=None, Ez=None, Ai=None, Aj=None, Av=None, b=None, coord_len=None, index_offset=0):
        super().__init__()
        self.index_offset = index_offset
        self.epsilon = torch.as_tensor(epsilon, dtype=torch.float64)
        self.coord = torch.as_tensor(coord, dtype=torch.float64)
        self.Ez = torch.as_tensor(Ez, dtype=torch.complex128)
        self.Ai = torch.as_tensor(Ai, dtype=torch.int64)  # (B, nnz)
        self.Aj = torch.as_tensor(Aj, dtype=torch.int64)
        self.Av = torch.as_tensor(Av, dtype=torch.complex128)
        self.b = torch.as_tensor(b, dtype=torch.complex128)  # (B, M)
        self.coord_len = torch.as_tensor(coord_len, dtype=torch.int64)  # (B, M)
    def __getitem__(self, index):
        global_index = self.index_offset + index
        epsilon = self.epsilon[index]  # (M,)
        coord = self.coord[index]  # (M, 2)
        ez = self.Ez[index]  # (M, 2)

        ai = self.Ai[index]  # (nnz,)
        aj = self.Aj[index]  # (nnz,)
        av = self.Av[index]

        b = self.b[index]  # (M,)
        coord_len = self.coord_len[index]  # (1,)
        return global_index, epsilon, coord, ez, ai, aj, av, b, coord_len

    def __len__(self):
        return len(self.epsilon)

# Usage in main
if __name__ == "__main__":
    # Load the data
    data_set = loadmat(DATA_DIR / "deepOnet_data_A_1558.mat")

    Epsilon_train = data_set['Eplison_train']  # (390, 64, 64)
    X_train = data_set['X_train']  # (4096, 2)
    Ez_train = data_set['Ez_train']  # (390, 4096, 2)

    Epsilon_test = data_set['Eplison_test']  # (98, 64, 64)
    X_test = data_set['X_test']  # (4096, 2)
    Ez_test = data_set['Ez_test']  # (98, 4096, 2)

    coord_len_train = data_set['coord_len_train']
    coord_len_test = data_set['coord_len_test']

    Ai_train, Aj_train = data_set['Ai_train'], data_set['Aj_train']
    Ai_test, Aj_test = data_set['Ai_test'], data_set['Aj_test']
    Av_train, Av_test = data_set['Av_train'], data_set['Av_test']
    b_train, b_test = data_set['b_train'], data_set['b_test']

    print(f"Train shapes: ε {Epsilon_train.shape}, X {X_train.shape}, Ez {Ez_train.shape}")

    # Prepare the dataset instances
    Train_dataset = GetDataset(Epsilon_train, X_train, Ez_train, Ai_train, Aj_train, Av_train, b_train, coord_len_train)
    Test_dataset = GetDataset(Epsilon_test, X_test, Ez_test, Ai_test, Aj_test, Av_test, b_test, coord_len_test)

    # Example DataLoader (first item is global_index)
    loader = DataLoader(Train_dataset, batch_size=4, shuffle=True)
    for indices, epsilon_data, coord_data, E_true, Ai, Aj, Av, b, coord_len in loader:
        print(f'indices: {indices}, B_eps shape: {epsilon_data.shape}, T_xy shape: {coord_data.shape}, Ez shape: {E_true.shape}')
        break
