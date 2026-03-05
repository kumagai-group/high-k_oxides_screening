import os
import json
from pathlib import Path
import numpy as np
from torch_geometric.loader import DataLoader
from common.utils import get_eigvals, irreps2tensor, tensor2irreps
from model.utils import validate
from model.params import DatasetParams, E3NNHyperParams, TrainingParams
from model.model import E3NNModel
from model.parity_plot import ParityPlot
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class BestPrediction:
    def __init__(
        self, 
        model: E3NNModel, 
        loader: DataLoader, 
        params_model: E3NNHyperParams, 
        params_train: TrainingParams, 
        params_data: DatasetParams, 
        dir_save: Path, 
    ):
        self.model = model
        self.model.eval()
        self.loader = loader
        
        self.params_model = params_model
        self.params_train = params_train
        self.params_data = params_data

        self.dir_save = dir_save
        os.makedirs(self.dir_save, exist_ok=True)
        self.path_save_irreps = self.dir_save / "irreps.npz"
        self.path_save_tensor = self.dir_save / "tensor.npz"
        self.path_save_eigs = self.dir_save / "eigs.npz"


    def predict(self):
        self.loss, self.dict_irreps = validate(
            model=self.model, 
            valid_loader=self.loader, 
            loss_func=self.params_train.get_loss_func(), 
            params_data=self.params_data, 
            is_test=True, 
            c=self.params_model.c, 
        )

        self.matname_all = self.dict_irreps["matname"]
        self.irreps_true_all = self.dict_irreps["true"]
        self.irreps_pred_all = self.dict_irreps["pred"]
        
        num_irreps = 9 if self.params_data.target == "born" else 6
        
        irreps_true_all, irreps_pred_all = [], []
        self.tensor_true_all, self.tensor_pred_all = [], []
        self.eigs_true_all, self.eigs_pred_all = [], []
        for matname, irreps_true, irreps_pred in zip(self.matname_all, self.irreps_true_all, self.irreps_pred_all):
            irreps_true_all.append(irreps_true.numpy().reshape(-1, num_irreps))
            irreps_pred_all.append(irreps_pred.numpy().reshape(-1, num_irreps))
            
            tensor_true = irreps2tensor(irreps_true).numpy().reshape(-1, 3, 3)
            tensor_pred = irreps2tensor(irreps_pred).numpy().reshape(-1, 3, 3)
            self.tensor_true_all.append(tensor_true)
            self.tensor_pred_all.append(tensor_pred)
            
            eigs_true = get_eigvals(tensor_true)
            eigs_pred = get_eigvals(tensor_pred)
            self.eigs_true_all.append(eigs_true)
            self.eigs_pred_all.append(eigs_pred)
        
        self.irreps_true_all = np.concatenate(irreps_true_all)
        self.irreps_pred_all = np.concatenate(irreps_pred_all)

        self.tensor_true_all = np.concatenate(self.tensor_true_all)
        self.tensor_pred_all = np.concatenate(self.tensor_pred_all)
        
        self.eigs_true_all = np.concatenate(self.eigs_true_all)
        self.eigs_pred_all = np.concatenate(self.eigs_pred_all)

        # print(self.irreps_true_all.shape, self.tensor_true_all.shape, self.eigs_true_all.shape)
        
        np.savez(self.path_save_irreps, matname=self.matname_all, true=self.irreps_true_all, pred=self.irreps_pred_all)
        np.savez(self.path_save_tensor, matname=self.matname_all, true=self.tensor_true_all, pred=self.tensor_pred_all)
        np.savez(self.path_save_eigs, matname=self.matname_all, true=self.eigs_true_all, pred=self.eigs_pred_all)


class Evaluation:
    def __init__(
        self, 
        params_model: E3NNHyperParams, 
        params_train: TrainingParams, 
        params_data: DatasetParams, 
        dir_save: Path, 
    ):
        self.params_model = params_model
        self.params_train = params_train
        self.params_data = params_data

        self.dir_save = dir_save
        self.path_save_irreps = self.dir_save / "irreps.npz"
        self.path_save_tensor = self.dir_save / "tensor.npz"
        self.path_save_eigs = self.dir_save / "eigs.npz"

        if self.params_data.target == "born":
            self.cmap = "Greens"
            self.color = "green"
        if self.params_data.target == "eled":
            self.cmap = "Oranges"
            self.color = "orange"
        if self.params_data.target == "iond":
            self.cmap = "Blues"
            self.color = "blue"
        
        self.dict_metrics = {
            "irreps": {
                "linear": {}, 
                "log": {}, 
            }, 
            "tensor": {
                "linear": {}, 
                "log": {}, 
            }, 
            "eigs": {
                "linear": {}, 
                "log": {}, 
            }, 
        }


    def load(self):
        assert self.path_save_irreps.exists()
        assert self.path_save_tensor.exists()
        assert self.path_save_eigs.exists()
        
        self.dict_irreps = dict(np.load(self.path_save_irreps))
        self.dict_tensor = dict(np.load(self.path_save_tensor))
        self.dict_eigs = dict(np.load(self.path_save_eigs))
    
    def calc_metrics(self, x, y):
        return {
            "r2": r2_score(x, y), 
            "mae": mean_absolute_error(x, y), 
            "rmse": np.sqrt(mean_squared_error(x, y)), 
        }
    
    def save_metrics(self):
        with open(self.dir_save / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.dict_metrics, f, ensure_ascii=False, indent=2)
    
    def eval_irreps(self):
        assert self.dict_irreps

        matname_all = self.dict_irreps["matname"]
        irreps_true_all = self.dict_irreps["true"]
        irreps_pred_all = self.dict_irreps["pred"]
        
        l0_true = irreps_true_all[:, 0]
        l0_pred = irreps_pred_all[:, 0]
        self.dict_metrics["irreps"]["linear"]["l0"] = self.calc_metrics(l0_true, l0_pred)
        if self.params_data.target in ["eled", "iond"]:
            l0_true_log = np.log10(l0_true[l0_pred > 0])
            l0_pred_log = np.log10(l0_pred[l0_pred > 0])
            self.dict_metrics["irreps"]["log"]["l0"] = self.calc_metrics(l0_true_log, l0_pred_log)
        
        if irreps_true_all.shape[1] == 9:
            l1_true = irreps_true_all[:, [1, 2, 3]].reshape(-1, )
            l1_pred = irreps_pred_all[:, [1, 2, 3]].reshape(-1, )
            self.dict_metrics["irreps"]["linear"]["l1"] = self.calc_metrics(l1_true, l1_pred)

            l2_true = irreps_true_all[:, [4, 5, 6, 7, 8]].reshape(-1, )
            l2_pred = irreps_pred_all[:, [4, 5, 6, 7, 8]].reshape(-1, )
            self.dict_metrics["irreps"]["linear"]["l2"] = self.calc_metrics(l2_true, l2_pred)

            Q_true = [l0_true, l1_true, l2_true]
            Q_pred = [l0_pred, l1_pred, l2_pred]
            filenames = ["pp_irrepl0.png", "pp_irrepl1.png", "pp_irrepl2.png"]

        else:
            l1_true = None
            l1_pred = None

            l2_true = irreps_true_all[:, [1, 2, 3, 4, 5]].reshape(-1, )
            l2_pred = irreps_pred_all[:, [1, 2, 3, 4, 5]].reshape(-1, )
            self.dict_metrics["irreps"]["linear"]["l2"] = self.calc_metrics(l2_true, l2_pred)

            Q_true = [l0_true, l2_true]
            Q_pred = [l0_pred, l2_pred]
            filenames = ["pp_irrepl0.png", "pp_irrepl2.png"]
        
        for data_true, data_pred, filename in zip(
            Q_true, 
            Q_pred, 
            filenames, 
        ):
            pp = ParityPlot(x_data=data_true, y_data=data_pred)
            pp.plot_without_hist(
                cmap=self.cmap, 
                color=self.color, 
                path_save=self.dir_save / filename, 
                show_2nd_axis=False, 
            )
            
            if self.params_data.target in ["eled", "iond"] and "l0" in filenames:
                data_log_true = np.log10(data_true)
                data_log_pred = np.log10(data_pred)
                
                data_log_true = data_log_true[data_log_pred > 0]
                data_log_pred = data_log_pred[data_log_pred > 0]
                
                pp_log = ParityPlot(x_data=data_log_true, y_data=data_log_pred)
                pp_log.plot_without_hist(
                    cmap=self.cmap, 
                    color=self.color, 
                    path_save=self.dir_save / filename.replace(".png", "_log.png"), 
                    show_2nd_axis=True, 
                )
    

    def eval_tensor(self):
        assert self.dict_tensor
        
        diag_true, diag_pred = [], []
        nondiag_true, nondiag_pred = [], []
        for k, dict_v in self.dict_tensor.items():
            pass
        
    def eval_eigs(self):
        assert self.dict_eigs
        
        self.dict_metrics_eigs = {}

        matname_all = self.dict_eigs["matname"]
        eigs_true_all = self.dict_eigs["true"]
        eigs_pred_all = self.dict_eigs["pred"]
        
        eigs_log_true_all, eigs_log_pred_all = [], []
        for matname, eigs_true, eigs_pred in zip(matname_all, eigs_true_all, eigs_pred_all):
            if self.params_data.target in ["eled", "iond"]:
                if np.any(eigs_pred <= 0):
                    print("pass:", matname)
                    print(eigs_true)
                    print(eigs_pred)

                    eigs_true = eigs_true[eigs_pred > 0]
                    eigs_pred = eigs_pred[eigs_pred > 0]
                eigs_log_true = np.log10(eigs_true)
                eigs_log_pred = np.log10(eigs_pred)

                # print(eigs_log_true.shape)

                eigs_log_true_all.append(eigs_log_true)
                eigs_log_pred_all.append(eigs_log_pred)
        
        # print("eigs_true_all.shape: ", eigs_true_all.shape)
        # print("eigs_pred_all.shape: ", eigs_pred_all.shape)
        
        eigs_true_all = eigs_true_all.reshape(-1, )
        eigs_pred_all = eigs_pred_all.reshape(-1, )

        self.dict_metrics["eigs"]["linear"] = self.calc_metrics(eigs_true_all, eigs_pred_all)
        pp = ParityPlot(x_data=eigs_true_all, y_data=eigs_pred_all)

        fixed_range = None
        fixed_ticks = None
        ticks_width = None
        fixed_range_log = None
        fixed_ticks_log = None
        ticks_width_log = None
        if self.params_data.target == "born":
            fixed_range = (-10, 15)
            fixed_ticks = (-10, -5, 0, 5, 10, 15)
            ticks_width = 5
        elif self.params_data.target == "eled":
            fixed_range_log = (0, 1.4)
            fixed_ticks_log = (0, 1.4)
            ticks_width_log = 0.2
        elif self.params_data.target == "iond":
            fixed_range_log = (-1, 3)
            fixed_ticks_log = (-1, 3)
            ticks_width_log = 1
        
        pp.plot_without_hist(
            cmap=self.cmap, 
            color=self.color, 
            path_save=self.dir_save / "pp_eigs.png", 
            show_2nd_axis=False, 
            size=50, 
            alpha=0.5, 
            fixed_range=fixed_range, 
            fixed_ticks=fixed_ticks, 
            ticks_width=ticks_width, 
        )
        
        if self.params_data.target in ["eled", "iond"]:
            eigs_log_true_all = np.concatenate(eigs_log_true_all)
            eigs_log_pred_all = np.concatenate(eigs_log_pred_all)
            
            eigs_log_true_all = eigs_log_true_all.reshape(-1, )
            eigs_log_pred_all = eigs_log_pred_all.reshape(-1, )
            
            self.dict_metrics["eigs"]["log"] = self.calc_metrics(eigs_log_true_all, eigs_log_pred_all)
            pp_log = ParityPlot(x_data=eigs_log_true_all, y_data=eigs_log_pred_all)
            pp_log.plot_without_hist(
                cmap=self.cmap, 
                color=self.color, 
                path_save=self.dir_save / "pp_eigs_log.png", 
                show_2nd_axis=True, 
                size=50, 
                alpha=0.5, 
                fixed_range=fixed_range_log, 
                fixed_ticks=fixed_ticks_log, 
                ticks_width=ticks_width_log, 
            )
