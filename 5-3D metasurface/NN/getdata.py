from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "samples and post proceeding" / "mat_data"


class GetDataset(Dataset):
    """
    Supports multiple input layouts:
    - Mode 1 (3-branch): epsilon, Ebz, and Lambda all provided
    - Mode 2 (2-branch): epsilon and Lambda; Ebz=None
    - Mode 3 (1-branch): Lambda only; epsilon=None, Ebz=None
    """

    def __init__(
        self,
        epsilon=None,
        coord=None,
        Ez=None,
        Ebz=None,
        Lambda=None,
        Ai=None,
        Aj=None,
        Av=None,
        b=None,
        coord_len=None,
        index_offset=0,
    ):
        super().__init__()
        self.index_offset = index_offset
        self.epsilon = torch.as_tensor(epsilon, dtype=torch.float64) if epsilon is not None else None
        self.coord = torch.as_tensor(coord, dtype=torch.float64)
        self.Ez = torch.as_tensor(Ez, dtype=torch.complex128)
        self.Ebz = torch.as_tensor(Ebz, dtype=torch.float64) if Ebz is not None else None
        self.Lambda = torch.as_tensor(Lambda, dtype=torch.float64) if Lambda is not None else None
        self.Ai = torch.as_tensor(Ai, dtype=torch.int64)
        self.Aj = torch.as_tensor(Aj, dtype=torch.int64)
        self.Av = torch.as_tensor(Av, dtype=torch.complex128)
        self.b = torch.as_tensor(b, dtype=torch.complex128)
        self.coord_len = torch.as_tensor(coord_len, dtype=torch.int64)

    def __len__(self):
        return len(self.coord)

    def __getitem__(self, index):
        global_index = self.index_offset + index
        epsilon = (
            self.epsilon[index]
            if self.epsilon is not None
            else torch.zeros(1, dtype=torch.float64)
        )
        coord = self.coord[index]
        ez = self.Ez[index]
        ebz = self.Ebz[index] if self.Ebz is not None else torch.zeros(1, dtype=torch.float64)
        lambda_data = self.Lambda[index] if self.Lambda is not None else None

        ai = self.Ai[index]
        aj = self.Aj[index]
        av = self.Av[index]
        b = self.b[index]
        coord_len = self.coord_len[index]
        return global_index, epsilon, coord, ez, ebz, lambda_data, ai, aj, av, b, coord_len


if __name__ == "__main__":
    data_set = loadmat(DATA_DIR / "deepOnet_data_3Dcase3_261.mat")

    Epsilon_train = data_set["Eplison_train"]
    X_train = data_set["X_train"]
    Ez_train = data_set["Ez_train"]

    Epsilon_test = data_set["Eplison_test"]
    X_test = data_set["X_test"]
    Ez_test = data_set["Ez_test"]

    Lambda_train = data_set["Lambda_train"]
    Lambda_test = data_set["Lambda_test"]

    coord_len_train = data_set["coord_len_train"]
    coord_len_test = data_set["coord_len_test"]

    Ai_train, Aj_train = data_set["Ai_train"], data_set["Aj_train"]
    Ai_test, Aj_test = data_set["Ai_test"], data_set["Aj_test"]
    Av_train, Av_test = data_set["Av_train"], data_set["Av_test"]
    b_train, b_test = data_set["b_train"], data_set["b_test"]

    print(f"Train shapes: ε {Epsilon_train.shape}, X {X_train.shape}, Ez {Ez_train.shape}, Lambda {Lambda_train.shape}")

    Train_dataset = GetDataset(
        Epsilon_train,
        X_train,
        Ez_train,
        None,
        Lambda_train,
        Ai_train,
        Aj_train,
        Av_train,
        b_train,
        coord_len_train,
    )
    Test_dataset = GetDataset(
        Epsilon_test,
        X_test,
        Ez_test,
        None,
        Lambda_test,
        Ai_test,
        Aj_test,
        Av_test,
        b_test,
        coord_len_test,
    )

    loader = DataLoader(Train_dataset, batch_size=4, shuffle=True)
    for indices, epsilon_data, coord_data, E_true, ebz_data, lambda_data, Ai, Aj, Av, b, coord_len in loader:
        print(
            f"indices: {indices}, eps: {epsilon_data.shape}, coord: {coord_data.shape}, "
            f"ez: {E_true.shape}, lambda: {lambda_data.shape if lambda_data is not None else None}"
        )
        break
