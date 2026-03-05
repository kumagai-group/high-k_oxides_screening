import json
import os
import random
import shutil
from typing import Dict, List
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from e3nn.o3 import Irreps
from model.params import TrainingParams, E3NNHyperParams, DatasetParams
from model.cross_validation import cross_validate
from dotenv import load_dotenv
load_dotenv()
SEED = int(os.getenv('SEED'))

import warnings
warnings.simplefilter("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default="none", help='prediction target: eled, iond, born')

    parser.add_argument('--n_fold', type=int, default=5)
    
    parser.add_argument('--datasize', type=int, default=9999)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--min_epoch', type=int, default=60, help='early epochs to enforce model checkpointing')
    parser.add_argument('--es_count', type=int, default=50)
    parser.add_argument('--init_lr', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.95, help='ratio for lr decay')
    parser.add_argument('--step_size', type=float, default=30, help='step size for lr decay')
    
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--dim_l0', type=int, default=64)
    parser.add_argument('--dim_l1', type=int, default=16)
    parser.add_argument('--dim_l2', type=int, default=32)
    parser.add_argument('--dim_edge', type=int, default=32)
    args = parser.parse_args()

    target = args.target
    assert target in ["eled", "iond", "born"]
    print(f"===== Start {target} prediction =====")

    dir_data = Path("./database")
    dir_st = dir_data / "st_pbesol"
    dir_diele = dir_data / "dielectric_pbesol"
    print("Structure dir: ", dir_st)
    print("Dielectrics dir: ", dir_diele)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_save = Path("results/cross_valid") / target / timestamp
    os.makedirs(dir_save, exist_ok=True)
    print("Save dir: ", dir_save)
    path_checkpoint = dir_save / "checkpoint.pth.tar"
    path_modelbest = dir_save / "model_best.pth.tar"

    with open(dir_save / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    is_site_pred = (target == "born")

    irreps_hid = Irreps(f"{args.dim_l0}x0e + {args.dim_l1}x1o + {args.dim_l1}x1e + {args.dim_l2}x2e")
    if target == "born":
        irreps_out = Irreps("1x0e + 1x1e + 1x2e")
    else:
        irreps_out = Irreps("1x0e + 1x2e")
    print("Output irreps: ", irreps_out)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    params_data = DatasetParams(
        target=target,
        dir_st=dir_st,
        dir_prop=dir_diele,
        datasize=args.datasize,
        r_cut=5.,
        dir_save=dir_save, 
    )
    params_data.save_params()
    dataset = params_data.g_data()
    print("len(dataset): ", len(dataset))

    params_model = E3NNHyperParams(
        n_conv=args.n_layer,
        l_max=2,
        activation="SiLU",
        irreps_hid=irreps_hid,
        irreps_out=irreps_out,
        dim_edge=args.dim_edge,
        is_site_pred=is_site_pred,
        output_allsite=False,
        dir_save=dir_save,
        # device_ids=[0, 1, 2, 3],
        device_ids=[0, 1],
        c="cuda",
    )
    params_model.save_params()

    params_train = TrainingParams(
        batch_size=args.batch_size,
        loss_func="MSE",
        max_epoch=args.max_epoch,
        min_epoch=args.min_epoch,
        init_lr=args.init_lr,
        min_lr=1e-10,
        gamma=args.gamma,
        step_size=args.step_size,
        es_count=args.es_count,
        n_folds=args.n_fold,
        path_checkpoint=path_checkpoint,
        path_modelbest=path_modelbest,
        dir_save=dir_save,
        seed=SEED, 
    )
    params_train.save_params()

    score_tmp = cross_validate(
        dataset=dataset,
        params_data=params_data,
        params_model=params_model,
        params_train=params_train,
    )

    print("cv score : ", score_tmp)


    dir_best = Path("results/cross_valid") / params_data.target / "best"
    if dir_best.exists():
        score_best = []
        for i in range(1, args.n_fold+1):
            with open(dir_best / f"fold{i}" / "test" /"metrics.json", "r", encoding="utf-8") as f:
                metrics = json.load(f)
            if params_data.target in ["eled", "iond"]:
                score_best.append(metrics["eigs"]["log"]["r2"])
            else:
                score_best.append(metrics["eigs"]["linear"]["r2"])
        score_best = np.mean(score_best)
    else:
        os.makedirs(dir_best, exist_ok=True)
        score_best = float("-inf")
    
    if score_tmp > score_best:
        shutil.copytree(dir_save, dir_best, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
