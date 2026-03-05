import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import math
import os
import torch.nn as nn
from e3nn.io import CartesianTensor

warnings.simplefilter('ignore')

EPSILON = 1e-8


def to_gpu(x, c='cuda'):
    if torch.cuda.is_available():
        x = x.to(c)
    return x


def get_eigvals(x:np.ndarray) -> np.ndarray:
    eigs, _ = np.linalg.eigh(x)
    return eigs


def tensor2irreps(tensor: torch.Tensor, target: str) -> torch.Tensor:
    assert target in ["born", "eled", "iond"]
    
    ct_sym = CartesianTensor("ij=ji")
    ct_asym = CartesianTensor("ij=-ji")
    irreps_sym = ct_sym.from_cartesian(tensor)  # (N,6)
    irreps_asym = ct_asym.from_cartesian(tensor)  # (N,3)
    
    if target in ["eled", "iond"]:
        return irreps_sym
    else:
        return torch.cat((irreps_sym[..., :1], irreps_asym, irreps_sym[..., 1:]), dim=-1)


def irreps2tensor(irreps: torch.Tensor) -> torch.Tensor:
    ct_sym = CartesianTensor("ij=ji")
    ct_asym = CartesianTensor("ij=-ji")

    if irreps.shape[-1] == 9:
        irreps_sym_head = torch.cat([irreps[..., :1], irreps[..., 4:]], dim=-1)  # (N,6)
        irreps_asym = irreps[..., 1:4]  # (N,3)

        tensor_sym = ct_sym.to_cartesian(irreps_sym_head)  # (N,3,3)
        tensor_asym = ct_asym.to_cartesian(irreps_asym)  # (N,3,3)
    else:
        tensor_sym = ct_sym.to_cartesian(irreps)  # (N,3,3)
        tensor_asym = torch.zeros_like(tensor_sym).to(tensor_sym.device)  # (N,3,3)

    return tensor_sym + tensor_asym


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, checkpoint_path, patience=5, verbose=False):
        self.patience = patience  # 設定ストップカウンタ
        self.verbose = verbose  # 表示の有無
        self.counter = 0  # 現在のカウンタ値
        self.best_score = None  # ベストスコア
        self.early_stop = False  # ストップフラグ
        self.val_loss_min = float("inf")  # 前回のベストスコア記憶用
        self.path = checkpoint_path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:  # 1Epoch目の処理
            self.best_score = score  # 1Epoch目はそのままベストスコアとして記録する
            # self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1  # ストップカウンタを+1
            if self.verbose:  # 表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  # 現在のカウンタを表示する
            if self.counter >= self.patience:  # 設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  # ベストスコアを更新した場合
            self.best_score = score  # ベストスコアを上書き
            # self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  # ストップカウンタリセット


def get_train_val_test_loader(
    dataset, 
    collate_fn=default_collate, 
    batch_size=64,
    train_ratio=None, 
    val_ratio=0.1, 
    test_ratio=0.1,
    return_test=False, 
    num_workers=1, 
    pin_memory=False, 
    device="cpu", 
):
    total_size = len(dataset)
    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
        print(
            f'[Warning] train_ratio is None, using 1 - val_ratio - '
            f'test_ratio = {train_ratio} as training data.'
        )
    else:
        assert train_ratio + val_ratio + test_ratio <= 1

    indices = list(range(total_size))
    random.shuffle(indices)
    train_size = int(train_ratio * total_size)
    test_size = int(test_ratio * total_size)
    valid_size = int(val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    else:
        test_sampler = None
    if device != torch.device("cpu"):
        num_workers = 8
        pin_memory = True
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn, 
        pin_memory=pin_memory, 
    )
    val_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn, 
        pin_memory=pin_memory
    )
    if return_test:
        test_loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn, pin_memory=pin_memory
        )

        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def save_checkpoint(state, is_best, checkpoint_path, modelbest_path):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, modelbest_path)


def show_scatter(df, plot_range, save_dir, filename="test_results_scatter", device=torch.device("cpu")):
    target_list = []
    pred_list = []
    for i in range(len(df)):
        target_path = df.iloc[i]["targets_ten_path"]
        pred_path = df.iloc[i]["preds_ten_path"]
        target = np.loadtxt(target_path)
        pred = np.loadtxt(pred_path)

        # 右上のみ
        for j in range(3):
            for k in range(j, 3):
                target_list.append(target[j][k])
                pred_list.append(pred[j][k])
    df_scatter = pd.DataFrame({"target": target_list, "pred": pred_list})
    df_scatter.to_csv(f"{save_dir}/{filename}.csv")
    fig = plt.figure()
    plt.plot(plot_range, plot_range)
    plt.scatter(target_list, pred_list)
    plt.savefig(f"{save_dir}/{filename}.png")
    plt.close()


def show_learning_curve(total_epochs, loss_train_list, loss_valid_list, save_dir,
                        filename="learning_curve", device=torch.device("cpu")):
    x = torch.arange(1, total_epochs + 1)

    df_out = pd.DataFrame({"epoch": x, "loss_train": loss_train_list, "loss_valid": loss_valid_list})
    df_out.to_csv(f"{save_dir}/{filename}.csv", index=False)


def show_error_deposition(dataframe, save_dir, filename="error_deposition", device=torch.device("cpu")):
    dataframe.sort_values("error_ten", ascending=True, inplace=True)
    x_count = []
    y_error = []
    for i in range(len(dataframe)):
        x_count.append(i)
        y_error.append(dataframe.iloc[i]["error_ten"])

    df_out = pd.DataFrame({"x_count": x_count, "y_error": y_error})
    df_out.to_csv(f"{save_dir}/{filename}.csv")

    fig = plt.figure()
    plt.scatter(x_count, y_error)
    plt.savefig(f"{save_dir}/{filename}.png")
    plt.close()


def show_function(dataframe, min_x, max_x, save_dir, filename="function.png", dim=256, device=torch.device("cpu")):
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    energy = np.linspace(min_x, max_x, dim)
    for i in range(min(100, len(dataframe))):
        row = i // 10
        col = i % 10

        df_ = dataframe.iloc[i]
        id_ = df_["id"]
        target_path = df_["target_path"]
        pred_path = df_["pred_path"]
        error = df_["error"]
        x = energy
        target_y = np.loadtxt(target_path)
        pred_y = np.loadtxt(pred_path)

        ax = axes[row, col]
        ax.plot(x, target_y)
        ax.plot(x, pred_y)
        ax.set_title(f"{id_}\n{error}")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}")
    plt.close()

def show_l0_scatter(df, save_dir, filename="test_l0_scatter", idx=0):
    material_id_list = []
    target_eigval_list = []
    pred_eigval_list = []
    for i in range(len(df)):
        material_id = df.iloc[i]["material_id"]
        targets_irrep_path = df.iloc[i]["targets_irrep_path"]
        preds_irrep_path = df.iloc[i]["preds_irrep_path"]

        target = np.loadtxt(targets_irrep_path)
        pred = np.loadtxt(preds_irrep_path)

        material_id_list.append(material_id)
        target_eigval_list.append(float(target[idx]))
        pred_eigval_list.append(float(pred[idx]))

    df_out = pd.DataFrame(
        {"material_id": material_id_list, "target": target_eigval_list, "pred": pred_eigval_list})
    df_out.to_csv(f"{save_dir}_{filename}.csv", index=False)


def show_eigvals_scatter(df, save_dir, filename=""):
    material_id_list = []
    target_eigval_list = []
    pred_eigval_list = []
    for i in range(len(df)):
        material_id = df.iloc[i]["material_id"]
        targets_ten_path = df.iloc[i]["targets_ten_path"]
        preds_ten_path = df.iloc[i]["preds_ten_path"]

        target = np.loadtxt(targets_ten_path)
        pred = np.loadtxt(preds_ten_path)

        target_eigvals = np.linalg.eigvals(target)
        pred_eigvals = np.linalg.eigvals(pred)

        for j, direction in enumerate(["xx", "yy", "zz"]):
            material_id_list.append(f"{material_id}_{direction}")
            target_eigval_list.append(float(target_eigvals[j]))
            pred_eigval_list.append(float(pred_eigvals[j]))

    df_out = pd.DataFrame(
        {"material_id": material_id_list, "target_eigval": target_eigval_list, "pred_eigval": pred_eigval_list})
    df_out.to_csv(f"{save_dir}_{filename}.csv", index=False)


def kkr(de, eps_imag, cshift=1e-1):
    nedos = eps_imag.size(1)
    cshift = complex(0, cshift)
    w_i_real = torch.arange(0, nedos * de, de, dtype=torch.float32)
    w_i = w_i_real.to(torch.complex64)
    w_i = w_i.view(1, nedos, 1, 1)

    def integration_element(w_r):
        # PyTorchでの計算
        factor = w_i / (w_i ** 2 - w_r ** 2 + cshift)
        total = torch.sum(eps_imag * factor, dim=1)
        return total * (2 / math.pi) * de + torch.diag(torch.tensor([1.0, 1.0, 1.0]))

    return torch.real(torch.stack([integration_element(w_r) for w_r in w_i[0, :, 0, 0]])).view(-1, nedos, 3, 3)


def calc_error(mae_or_mse, target, pred):
    target = np.array(target)
    pred = np.array(pred)
    if mae_or_mse == "mae":
        error = np.mean(np.abs(target - pred))
    elif mae_or_mse == "mse":
        error = np.mean(np.square(target - pred))
    else:
        error = 0.0

    return error


def init_before_training(base_dir):
    save_dir = f"{base_dir}/results"
    dataset_path = f"{base_dir}/tfn_dataset.pt"
    checkpoint_path = f"{save_dir}/checkpoint.pth.tar"
    modelbest_path = f"{save_dir}/model_best.pth.tar"

    print("Initializing Results dir")
    shutil.rmtree(save_dir)
    os.makedirs(f"{save_dir}/equivariance/preds")
    os.makedirs(f"{save_dir}/equivariance/r_preds")
    os.makedirs(f"{save_dir}/train_preds")
    os.makedirs(f"{save_dir}/train_targets")
    os.makedirs(f"{save_dir}/test_preds")
    os.makedirs(f"{save_dir}/test_targets")

    return save_dir, dataset_path, checkpoint_path, modelbest_path


class CustomLossForSpectra(nn.Module):
    def __init__(self):
        super().__init__()
        # 初期化処理
        # self.param = ...

    def forward(self, outputs, targets):
        # [N, F, 3, 3]

        # AREA
        area_outputs = torch.mean(outputs, dim=1, keepdim=True)
        area_targets = torch.mean(targets, dim=1, keepdim=True)
        diff_area = torch.abs(area_outputs - area_targets)
        diff_area = torch.mean(diff_area)

        # # PCC
        # mean_out = torch.mean(outputs, dim=1, keepdim=True)
        # mean_tar = torch.mean(targets, dim=1, keepdim=True)
        #
        # std_out = torch.sqrt(torch.sum((outputs - mean_out) ** 2, dim=1, keepdim=True))
        # std_tar = torch.sqrt(torch.sum((targets - mean_tar) ** 2, dim=1, keepdim=True))
        #
        # cov = torch.sum((outputs - mean_out) * (targets - mean_tar), dim=1, keepdim=True)
        #
        # pcc_mean = torch.mean((cov / ((std_out * std_tar) + EPSILON)))

        # MAE
        mae = torch.mean(torch.abs(outputs - targets))

        return mae + diff_area


class CustomLossForTensor(nn.Module):
    def __init__(self):
        super().__init__()
        # 初期化処理
        # self.param = ...

    def forward(self, outputs, targets):
        # [N, 1, 6]

        # [1, 1, 6]
        bias = torch.Tensor([1., 1., 1., 10., 1., 10.]).unsqueeze(0).unsqueeze(0)
        mae = torch.mean(torch.abs(outputs - targets) * bias)

        return mae


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        return torch.sqrt(self.mse(preds, targets))


class MAEandSignPenaLoss(nn.Module):
    def __init__(self, sign_pena_weight=1):
        super().__init__()
        print("criterion: MAEandSignPenaLoss")
        self.mae = nn.L1Loss()
        self.relu = nn.functional.relu

        self.sign_pena_weight = sign_pena_weight

    def forward(self, preds, targets):
        # [N, F, 6]
        sign_pena = self.relu(-1. * preds * targets)

        # print("mae: ", float(self.mae(preds, targets)), ", sign_pena: ", float(self.sign_pena_weight * torch.mean(sign_pena)))

        return self.mae(preds, targets) + self.sign_pena_weight * torch.square(torch.mean(sign_pena))


class SignPenaLoss(nn.Module):
    def __init__(self):
        super().__init__()
        print("criterion: SignPenaLoss")
        self.relu = nn.functional.relu

    def forward(self, preds, targets):
        preds = torch.div(preds, torch.abs(preds)+EPSILON)
        targets = torch.div(targets, torch.abs(targets)+EPSILON)

        # [N, F, 6]
        sign_pena = self.relu(-1. * preds * targets)

        return torch.mean(sign_pena)


def standardization(dataset, save_dir):
    # [N, F, 9]
    target = torch.cat([dataset[i][1] for i in range(len(dataset))], dim=0)

    mean_list, std_list = [], []
    for i in range(target.shape[-1]):
        # [N, F, 1]
        irrep = target[:, :, i].unsqueeze(-1)

        # [1, F, 1] ## スペクトルの場合どう標準化するか？
        mean = torch.mean(irrep, dim=0, keepdim=True)
        std = torch.std(irrep, dim=0, keepdim=True) + 1e-10

        mean_list.append(mean)
        std_list.append(std)

    # [1, F, 9]
    mean_list = torch.cat(mean_list, dim=-1)
    std_list = torch.cat(std_list, dim=-1)
    print("mean: ", mean_list.squeeze(), mean_list.shape)
    print("std: ", std_list.squeeze(), std_list.shape)

    for i in range(len(dataset)):
        dataset[i][1][:] = torch.div(dataset[i][1][:] - mean_list, std_list)

    std_target = torch.cat([dataset[i][1] for i in range(len(dataset))], dim=0)
    for i in range(target.shape[1]):
        df = pd.DataFrame({"0": target[:, i, 0], "1": target[:, i, 1], "2": target[:, i, 2],
                           "3": target[:, i, 3], "4": target[:, i, 4], "5": target[:, i, 5],
                           "std_0": std_target[:, i, 0], "std_1": std_target[:, i, 1], "std_2": std_target[:, i, 2],
                           "std_3": std_target[:, i, 3], "std_4": std_target[:, i, 4], "std_5": std_target[:, i, 5]
                           })
        df.to_csv(f"{save_dir}/target{i}_irrep.csv", index=False)

    return dataset, mean_list, std_list


def f_env(p, d):
    a = 1.
    b = torch.div((p+1.)*(p+2.), 2) * torch.Tensor(np.power(d, p))
    c = p*(p+2)*torch.Tensor(np.power(d, p+1))
    d = torch.div(p*(p+1), 2) * torch.Tensor(np.power(d, p+2))

    return a - b + c - d


def radial_bessel_func(dij, rc, b, add_to_a):
    # [13, 13, 1], [], [100]

    # [12, 1], [13, 13, 1]
    dij = dij.unsqueeze(-1)
    if add_to_a:
        # [1, 100]
        b = b.unsqueeze(0)
    else:
        # [1, 1, 100]
        b = b.unsqueeze(0).unsqueeze(0)

    # radial_bessel_out = torch.div(2 * torch.Tensor(np.sin(b * np.pi * dij / rc)), (rc * dij)) * f_env(6, dij)
    radial_bessel_out = torch.div(2 * torch.Tensor(np.sin(b * np.pi * dij / rc)), (rc * dij))

    return radial_bessel_out


def logistic_function(b, d, r=5.0, k=1.0, f=10, device=torch.device("cpu"), c='cuda:0'):

    exp = torch.exp(to_gpu(torch.Tensor(k * (d - r)), c=c))
    sin = torch.sin(to_gpu(torch.Tensor(2 * np.pi * b / f), c=c))

    return (1 / (1 + exp)) * sin


def get_criterion(error_type, sign_pena_weight):
    if error_type == "mae":
        criterion = nn.L1Loss()
    elif error_type == "mse":
        criterion = nn.MSELoss()
    elif error_type == "rmse":
        criterion = RMSELoss()
    elif error_type == "mae_and_sign_pena":
        criterion = MAEandSignPenaLoss(sign_pena_weight=sign_pena_weight)
    elif error_type == "sign_pena":
        criterion = SignPenaLoss()
    else:
        criterion = nn.L1Loss()

    return criterion


def matrix_log(t: torch.Tensor, base: float = 10.0) -> torch.Tensor:
    """
    対称行列 t (Tensor) の特異値分解を行い、その特異値の対数を取った行列を返す関数。
    base: 対数を取る際の底 (既定値10)
    """
    u, s, v = torch.linalg.svd(t)
    log_cov = torch.matmul(
        torch.matmul(u, torch.diag_embed(torch.log(s))),
        v
    )
    return log_cov * float(np.log(base))
