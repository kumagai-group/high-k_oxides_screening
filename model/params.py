from pathlib import Path
from typing import List
import torch
import torch.nn as nn
import numpy as np
from e3nn.o3 import Irreps
from pymatgen.core import Structure

from torch.optim.lr_scheduler import StepLR
import json
from model.graph import CrystalGraphDataset


class DatasetParams:
    def __init__(
        self,
        target: str, 
        dir_st: Path, 
        dir_prop: Path,
        datasize: int,
        r_cut: float,
        dir_save: Path,
):
        self.target = target
        self.dir_st = dir_st
        self.dir_prop = dir_prop
        self.datasize = datasize
        self.r_cut = r_cut
        self.dir_save = dir_save
    
    @classmethod
    def load_params(cls, dir_save):
        with open(dir_save / "params_dataset.json", "rb") as f:
            dp_dict = json.load(f)
            f.close()

        return cls(
            target=dp_dict["target"], 
            dir_st=Path(dp_dict["dir_st"]), 
            dir_prop = Path(dp_dict["dir_prop"]), 
            datasize = dp_dict["datasize"], 
            r_cut = dp_dict["r_cut"], 
            dir_save = Path(dp_dict["dir_save"]), 
        )

    def g_data(self):
        matnames = [str(p).split("/")[-1] for p in self.dir_st.iterdir() if p.is_dir()]
        self.dict_data = {}
        for matname in matnames:
            st = Structure.from_file(self.dir_st / matname / "st.cif")
            if self.target == "eled":
                prop = np.loadtxt(self.dir_prop/ matname / "eled.txt")
            elif self.target == "iond":
                prop = np.loadtxt(self.dir_prop / matname / "iond.txt")
            elif self.target == "born":
                prop = np.load(self.dir_prop / matname / "becs.npy")
                assert prop.shape[0] == len(st.sites)
            
            self.dict_data[matname] = {
                "structure": st, 
                "prop": torch.tensor(prop, dtype=torch.float32), 
            }

        self.dict_atom_init = self.load_dict_atom_init()

        dataset = CrystalGraphDataset(
            target=self.target,
            dict_data=self.dict_data,
            path_save=self.dir_save / "dataset.pth",
            datasize=self.datasize,
            rmax=self.r_cut,
            lmax=2,
            dict_atom_init=self.dict_atom_init,
        )

        return dataset


    def load_dict_atom_init(self):
        import json
        with open("database/atom_init.json") as f:
            elem_embedding = json.load(f)
        dict_atom_init = {int(key): value for key, value in elem_embedding.items()}

        return dict_atom_init


    def save_params(self):
        dict_dp = {
            "target": self.target,
            "dir_st": str(self.dir_st),
            "dir_prop": str(self.dir_prop),
            "datasize": self.datasize,
            "r_cut": self.r_cut,
            "dir_save": str(self.dir_save)
        }

        with open(self.dir_save / "params_dataset.json", "w") as f:
            json.dump(dict_dp, f, indent=2)
            f.close()


class TrainingParams:
    def __init__(
        self,
        batch_size: int, 
        loss_func: str,
        max_epoch: int,
        min_epoch: int, 
        init_lr: float,
        min_lr: float,
        gamma: float,
        step_size: int,
        es_count: int,
        n_folds: int,
        path_checkpoint: Path,
        path_modelbest: Path,
        dir_save: Path,
        seed: int, 
    ):
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.max_epoch = max_epoch
        self.min_epoch = min_epoch

        self.init_lr = init_lr
        self.min_lr = min_lr
        self.gamma = gamma
        self.step_size = step_size
        self.es_count = es_count

        self.n_folds = n_folds

        self.dir_save = dir_save
        self.path_checkpoint = path_checkpoint
        self.path_modelbest = path_modelbest

        self.seed = seed

    @classmethod
    def load_params(cls, dir_save):
        with open(dir_save / "params_training.json", "rb") as f:
            hp_dict = json.load(f)
            f.close()
        
        return cls(
            n_folds=hp_dict["n_folds"],
            batch_size=hp_dict["batch_size"],
            loss_func=hp_dict["loss_func"],
            max_epoch=hp_dict["max_epoch"],
            min_epoch=hp_dict["min_epoch"],
            init_lr=hp_dict["init_lr"],
            min_lr=hp_dict["min_lr"],
            gamma=hp_dict["gamma"],
            step_size=hp_dict["step_size"],
            es_count=hp_dict["es_count"], 
            path_checkpoint=Path(hp_dict["path_checkpoint"]),
            path_modelbest=Path(hp_dict["path_modelbest"]),
            dir_save=Path(hp_dict["dir_save"]),
            seed=hp_dict["seed"],
        )

    def get_loss_func(self):
        if self.loss_func == "MSE":
            return nn.MSELoss()
        else:
            return nn.L1Loss()
    
    def get_optimizer(self, model):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.init_lr, weight_decay=0.0)
        return self.optimizer

    def get_scheduler(self):
        if self.optimizer is None:
            raise ValueError("Optimizer not initialized. Call get_optimizer() first.")
        return StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def save_params(self):
        tp_dict = {
            "n_folds": self.n_folds, 
            "batch_size": self.batch_size,
            "loss_func": self.loss_func,
            "max_epoch": self.max_epoch,
            "min_epoch": self.min_epoch,
            "init_lr": self.init_lr,
            "min_lr": self.min_lr,
            "gamma": self.gamma,
            "step_size": self.step_size,
            "es_count": self.es_count,
            "path_checkpoint": str(self.path_checkpoint),
            "path_modelbest": str(self.path_modelbest),
            "dir_save": str(self.dir_save),
            "seed": self.seed,
        }

        with open(self.dir_save / "params_training.json", "w") as f:
            json.dump(tp_dict, f, indent=2)
            f.close()


class E3NNHyperParams:
    def __init__(
            self,
            n_conv: int,
            l_max: int,
            activation: str,
            irreps_hid: Irreps,
            irreps_out: Irreps,
            dim_edge: int,
            is_site_pred: bool,
            output_allsite: bool,
            dir_save: Path,
            device_ids: List[int],
            c: str="cuda",
    ):
        self.n_conv = n_conv
        self.l_max = l_max
        self.activation = activation
        
        self.dim_edge = dim_edge
        
        self.is_site_pred = is_site_pred
        self.output_allsite = output_allsite
        
        self.dir_save = dir_save
        
        self.device_ids = device_ids
        self.c = c

        self.irreps_in = Irreps("92x0e")
        self.irreps_sh = Irreps.spherical_harmonics(lmax=self.l_max)
        self.irreps_hid = irreps_hid
        self.irreps_out = irreps_out
    
    @classmethod
    def load_params(cls, dir_save):
        with open(dir_save / "params_model_hyper.json", "rb") as f:
            hp_dict = json.load(f)
            f.close()
        
        irreps_hid = Irreps(hp_dict["irreps_hid"])
        irreps_out = Irreps(hp_dict["irreps_out"])

        return cls(
            n_conv=3,
            l_max=hp_dict["l_max"],
            activation=hp_dict["activation"],
            irreps_hid=irreps_hid,
            irreps_out=irreps_out,
            dim_edge=hp_dict["dim_edge"],
            is_site_pred=hp_dict["is_site_pred"],
            output_allsite=hp_dict["output_allsite"],
            dir_save=Path(hp_dict["dir_save"]),
            device_ids=hp_dict["device_ids"],
            c=hp_dict["c"],
        )

    def get_act_func(self):
        if self.activation == "ReLU":
            return nn.ReLU()
        elif self.activation == "SiLU":
            return nn.SiLU()
        else:
            return nn.SiLU()

    # def get_model(self) -> nn.Module:
    #     self.model = E3NNModel(
    #         irreps_in=self.irreps_in,
    #         irreps_sh=self.irreps_sh,
    #         irreps_hid=self.irreps_hid,
    #         irreps_out=self.irreps_out,
    #         dim_edge=self.dim_edge,
    #         n_conv=self.n_conv,
    #         activation=self.get_act_func(),
    #         is_site_pred=self.is_site_pred,
    #         output_allsite=self.output_allsite, 
    #     )
    #     assert torch.cuda.is_available()
    #     self.model = DataParallel(self.model, device_ids=self.device_ids)
    #     self.model = to_gpu(self.model, c=self.c)

    #     trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    #     print("num tranable params:", trainable_params)

    #     for name, param in self.model.named_parameters():
    #         print(f"{name}: {param.numel()}")

    #     return self.model

    def save_params(self):
        hp_dict = {
            "l_max": self.l_max,
            "irreps_hid": str(self.irreps_hid),
            "irreps_out": str(self.irreps_out),
            "dim_edge": self.dim_edge,
            "n_conv": self.n_conv,
            "activation": self.activation,
            "is_site_pred": self.is_site_pred,
            "output_allsite": self.output_allsite,
            "dir_save": str(self.dir_save),
            "device_ids": self.device_ids,
            "c": self.c,
        }

        with open(self.dir_save / "params_model_hyper.json", "w") as f:
            json.dump(hp_dict, f, indent=2)
            f.close()
