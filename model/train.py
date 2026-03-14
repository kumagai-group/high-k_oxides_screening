import os
import warnings
from typing import Tuple, List
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import common.utils as com_utils
from model.utils import train, validate
from model.params import DatasetParams, E3NNHyperParams, TrainingParams
from dotenv import load_dotenv
load_dotenv()

warnings.simplefilter("ignore")

SEED = os.getenv('SEED')


def train_e3nn_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    params_data: DatasetParams,
    params_model: E3NNHyperParams,
    params_train: TrainingParams,
) -> Tuple[List[float], List[float], float]:
    optimizer = params_train.get_optimizer(model)
    scheduler = params_train.get_scheduler()

    best_valid_loss = 1e10
    path_checkpoint = params_train.path_checkpoint
    earlystopping = com_utils.EarlyStopping(path_checkpoint, patience=params_train.es_count, verbose=True)
    train_loss_list, valid_loss_list = [], []

    max_epoch = params_train.max_epoch
    for epoch in range(1, max_epoch + 1):
        train_loss = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_func=params_train.get_loss_func(),
            params_data=params_data, 
            c=params_model.c, 
        )
        train_loss_list.append(float(train_loss))

        valid_loss, _ = validate(
            model=model,
            valid_loader=valid_loader,
            loss_func=params_train.get_loss_func(),
            params_data=params_data, 
            c=params_model.c, 
        )
        valid_loss_list.append(float(valid_loss))

        print(f"[Epoch {epoch}/{max_epoch}] Train: {train_loss:.6f}, Valid: {valid_loss:.6f}")

        if epoch <= params_train.min_epoch:
            is_best = True
            best_valid_loss = valid_loss
        else:
            is_best = (valid_loss < best_valid_loss)
            best_valid_loss = min(valid_loss, best_valid_loss)

        com_utils.save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_mae_error": best_valid_loss,
                "optimizer": optimizer.state_dict(),
            }, 
            is_best, 
            path_checkpoint, 
            params_train.path_modelbest, 
        )

        if epoch > params_train.min_epoch:
            earlystopping(valid_loss, model)
        
        if earlystopping.early_stop:
            print("Early stopping triggered.")
            break

    return train_loss_list, valid_loss_list, best_valid_loss


def train_e3nn_model_full(
    model: nn.Module,
    full_loader: DataLoader,
    d_params: object,
    m_params: object,
    t_params: object,
) -> List[float]:
    optimizer, scheduler = t_params.get_opt_and_sch(model)
    loss_func = t_params.get_loss_func()

    train_loss_list = []

    max_epoch = t_params.max_epoch
    for epoch in range(1, max_epoch + 1):
        train_loss = train(
            model=model,
            train_loader=full_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_func=loss_func,
            datatype=t_params.datatype,
            modelname=m_params.modelname,
            c=m_params.c
        )
        train_loss_list.append(float(train_loss))
        print(f"[Epoch {epoch}/{max_epoch}] Train Loss: {train_loss:.6f}")

    # 最終モデルの保存
    checkpoint_path = t_params.checkpoint_path
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        "epoch": max_epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, checkpoint_path)
    print(f"モデルを保存しました: {checkpoint_path}")

    modelbest_path = t_params.modelbest_path
    os.makedirs(os.path.dirname(modelbest_path), exist_ok=True)
    torch.save({
        "epoch": max_epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, modelbest_path)
    print(f"モデルを保存しました: {modelbest_path}")

    return train_loss_list
