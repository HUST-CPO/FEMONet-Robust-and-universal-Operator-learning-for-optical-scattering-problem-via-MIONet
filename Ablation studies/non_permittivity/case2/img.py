import scipy.io as sio
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import re
import math


def calculate_gt_energy_stats(mat_path, batch_size=32):
    """Calculate gt energy stats."""
    data = sio.loadmat(mat_path)

    Ai_tr, Aj_tr, Av_tr = data["Ai_train"], data["Aj_train"], data["Av_train"]
    b_tr, Ez_tr = data["b_train"], data["Ez_train"]
    len_tr = data["coord_len_train"].flatten()
    num_train = Ez_tr.shape[0]

    train_total_energy = 0
    for i in range(num_train):
        Mi = int(len_tr[i])

        mask = (Ai_tr[i] > 0) & (Aj_tr[i] > 0)
        A = sp.csc_matrix(
            (Av_tr[i, mask], (Ai_tr[i, mask] - 1, Aj_tr[i, mask] - 1)), shape=(Mi, Mi)
        )
        u = Ez_tr[i, :Mi]
        b = b_tr[i, :Mi]

        e = 0.5 * np.dot(u, A.dot(u)) - np.dot(u, b)
        train_total_energy += e.real

    num_train_batches = math.ceil(num_train / batch_size)
    train_ref = train_total_energy / num_train_batches

    Ai_te, Aj_te, Av_te = data["Ai_test"], data["Aj_test"], data["Av_test"]
    b_te, Ez_te = data["b_test"], data["Ez_test"]
    len_te = data["coord_len_test"].flatten()
    num_test = Ez_te.shape[0]

    test_total_energy = 0
    for i in range(num_test):
        Mi = int(len_te[i])
        mask = (Ai_te[i] > 0) & (Aj_te[i] > 0)
        A = sp.csc_matrix(
            (Av_te[i, mask], (Ai_te[i, mask] - 1, Aj_te[i, mask] - 1)), shape=(Mi, Mi)
        )
        u = Ez_te[i, :Mi]
        b = b_te[i, :Mi]
        e = 0.5 * np.dot(u, A.dot(u)) - np.dot(u, b)
        test_total_energy += e.real

    test_ref = test_total_energy

    return train_ref, test_ref


def plot_residual_from_log(log_path, mat_path):

    train_ref, test_ref = calculate_gt_energy_stats(mat_path, batch_size=32)

    print("-" * 40)
    print(f"训练集理论参考能量 (Train Ref): {train_ref:.6f}")
    print(f"测试集理论参考能量 (Test Ref): {test_ref:.6f}")
    print("-" * 40)

    epochs, train_loss_list, test_loss_list = [], [], []
    pattern = re.compile(
        r"Epoch (\d+), Total Loss: ([-+]?\d*\.\d+|\d+), test Loss ([-+]?\d*\.\d+|\d+)"
    )

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                train_loss_list.append(float(match.group(2)))
                test_loss_list.append(float(match.group(3)))

    epochs = np.array(epochs)
    train_loss = np.array(train_loss_list)
    test_loss = np.array(test_loss_list)

    # Residual = |Loss_pred - Energy_GT| / |Energy_GT|
    train_res = np.abs(train_loss - train_ref) / np.abs(train_ref)
    test_res = np.abs(test_loss - test_ref) / np.abs(test_ref)

    plt.figure(figsize=(10, 6), dpi=120)
    plt.semilogy(
        epochs, train_res, label="train_loss_residual", color="#1f77b4", lw=1.5
    )
    plt.semilogy(epochs, test_res, label="test_loss_residual", color="#ff7f0e", lw=1.5)

    plt.title("Normalized Energy Residual: Neural Net vs. Ground Truth", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Relative Difference (Log Scale)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(frameon=True)

    plt.tight_layout()
    plt.savefig("energy_residual_recalculated.png")
    plt.show()


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument(
        "--data",
        default=str(Path(__file__).resolve().parent / "deepOnet_data_B1_847.mat"),
    )
    args = parser.parse_args()
    plot_residual_from_log(args.log, args.data)
