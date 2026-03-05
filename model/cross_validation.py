import os
import warnings
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from torch.utils.data.dataset import Subset, Dataset
import matplotlib.pyplot as plt
from model.construct_model import create_e3nn_model
from model.train import train_e3nn_model
from model.evaluation import BestPrediction, Evaluation
from model.params import DatasetParams, E3NNHyperParams, TrainingParams
from dotenv import load_dotenv
load_dotenv()

warnings.simplefilter("ignore")


class LossCurve:
    def __init__(
        self, loss_train: list, loss_valid: list, dir_save: Path, 
    ):
        self.loss_train = loss_train
        self.loss_valid = loss_valid
        self.path_save_csv = dir_save / "loss_curve.csv"
        self.path_save_img = dir_save / "loss_curve.png"
    
    def loss_curve(self):
        df_losscurve = pd.DataFrame(dict(train=self.loss_train, valid=self.loss_valid))
        df_losscurve.to_csv(self.path_save_csv, index=True)
        epochs = list(range(len(df_losscurve)))
        train_loss = df_losscurve["train"]
        valid_loss = df_losscurve["valid"]
        plt.scatter(epochs, train_loss)
        plt.scatter(epochs, valid_loss)
        plt.savefig(self.path_save_img)
        plt.close()


def cross_validate(
        dataset: Dataset,
        params_data: DatasetParams,
        params_model: E3NNHyperParams,
        params_train: TrainingParams,
) -> float:
    n_folds = params_train.n_folds

    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    fold_sizes = np.full(n_folds, len(dataset) // n_folds, dtype=int)
    fold_sizes[: len(dataset) % n_folds] += 1

    current = 0
    folds = []
    for fold in range(n_folds):
        start, stop = current, current + fold_sizes[fold]
        folds.append(indices[start:stop])
        current = stop
    # print("folds: ", folds)

    test_scores = []
    matname_allfolds = []
    irreps_true_allfolds, irreps_pred_allfolds = [], []
    tensor_true_allfolds, tensor_pred_allfolds = [], []
    eigs_true_allfolds, eigs_pred_allfolds = [], []
    for fold in range(n_folds):
        print(f"start fold: {fold}")
        test_idx = folds[fold]
        remaining_idx = list(set(indices) - set(test_idx))

        n_remaining = len(remaining_idx)
        n_valid = int(0.2 * n_remaining)
        valid_idx = remaining_idx[:n_valid]
        train_idx = remaining_idx[n_valid:]

        train_dataset = Subset(dataset, train_idx)
        valid_dataset = Subset(dataset, valid_idx)
        test_dataset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_dataset, batch_size=params_train.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=params_train.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=params_train.batch_size, shuffle=False)

        params_model.output_allsite = False
        model_cv = create_e3nn_model(e3nn_params=params_model)
        # model_cv = params_model.get_model()

        dir_cv = Path(params_train.dir_save) / f"fold{fold+1}"
        os.makedirs(dir_cv, exist_ok=True)
        params_train.path_checkpoint = dir_cv / "checkpoint.pth.tar"
        params_train.path_modelbest = dir_cv / "model_best.pth.tar"

        train_loss_list, valid_loss_list, _ = train_e3nn_model(
            model=model_cv,
            train_loader=train_loader,
            valid_loader=valid_loader,
            params_data=params_data,
            params_model=params_model,
            params_train=params_train,
        )

        lc = LossCurve(
            loss_train=train_loss_list,
            loss_valid=valid_loss_list,
            dir_save=dir_cv,
        )
        lc.loss_curve()

        best_checkpoint = torch.load(
            params_train.path_modelbest,
            map_location=params_model.c,
            weights_only=False
        )
        model_cv.load_state_dict(best_checkpoint["state_dict"])

        for key, loader in zip(
            ["train", "valid", "test"], 
            [train_loader, valid_loader, test_loader], 
        ):
            # print(f"# {key}")
            best_pred = BestPrediction(
                model=model_cv, 
                loader=loader, 
                params_model=params_model, 
                params_train=params_train, 
                params_data=params_data, 
                dir_save=dir_cv / key, 
            )
            best_pred.predict()
            if key == "test":
                matname_allfolds.extend(best_pred.matname_all)
                irreps_true_allfolds.append(best_pred.irreps_true_all)
                irreps_pred_allfolds.append(best_pred.irreps_pred_all)
                tensor_true_allfolds.append(best_pred.tensor_true_all)
                tensor_pred_allfolds.append(best_pred.tensor_pred_all)
                eigs_true_allfolds.append(best_pred.eigs_true_all)
                eigs_pred_allfolds.append(best_pred.eigs_pred_all)

        for key in ["train", "valid", "test"]:
            # print(f"# {key}")
            eval = Evaluation(
                params_model=params_model, 
                params_train=params_train, 
                params_data=params_data, 
                dir_save=dir_cv / key, 
            )
            eval.load()
            eval.eval_eigs()

            if key == "test":
                if params_data.target in ["eled", "iond"]:
                    test_scores.append(eval.dict_metrics["eigs"]["log"]["r2"])
                else:
                    test_scores.append(eval.dict_metrics["eigs"]["linear"]["r2"])
        eval.save_metrics()

    print("# allfolds")
    irreps_true_allfolds, irreps_pred_allfolds = np.concatenate(irreps_true_allfolds), np.concatenate(irreps_pred_allfolds)
    tensor_true_allfolds, tensor_pred_allfolds = np.concatenate(tensor_true_allfolds), np.concatenate(tensor_pred_allfolds)
    eigs_true_allfolds, eigs_pred_allfolds = np.concatenate(eigs_true_allfolds), np.concatenate(eigs_pred_allfolds)
    # print(irreps_true_allfolds.shape, irreps_pred_allfolds.shape)
    # print(tensor_true_allfolds.shape, tensor_pred_allfolds.shape)
    # print(eigs_true_allfolds.shape, eigs_pred_allfolds.shape)
    
    dir_allfolds = params_train.dir_save / "allfolds" / "test"
    os.makedirs(dir_allfolds, exist_ok=True)
    np.savez(dir_allfolds / "irreps.npz", matname=matname_allfolds, true=irreps_true_allfolds, pred=irreps_pred_allfolds)
    np.savez(dir_allfolds / "tensor.npz", matname=matname_allfolds, true=tensor_true_allfolds, pred=tensor_pred_allfolds)
    np.savez(dir_allfolds / "eigs.npz", matname=matname_allfolds, true=eigs_true_allfolds, pred=eigs_pred_allfolds)

    return np.mean(test_scores)
