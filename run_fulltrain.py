import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from model.params import TrainingParams, E3NNHyperParams, DatasetParams
from dotenv import load_dotenv
load_dotenv()
SEED = int(os.getenv('SEED'))

import warnings
warnings.simplefilter("ignore")

from torch.utils.data.dataset import Subset
from torch_geometric.loader import DataLoader
from model.construct_model import create_e3nn_model
from model.train import train_e3nn_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default="none", help='prediction target: eled, iond, born')
    args = parser.parse_args()
    
    target = args.target
    assert target in ["eled", "iond", "born"]
    print(f"===== Start {target} prediction =====")
    
    dir_best = Path("results/cross_valid") / target / "best"
    assert dir_best.exists()

    # load params
    params_data = DatasetParams.load_params(dir_best)
    params_train = TrainingParams.load_params(dir_best)
    params_model = E3NNHyperParams.load_params(dir_best)
    
    # set num epoch
    epochs = []
    for i_cv in range(1, params_train.n_folds+1):
        dir_cv = dir_best / f"fold{i_cv}"
        df_lc = pd.read_csv(dir_cv / "loss_curve.csv")
        if len(df_lc) == params_train.max_epoch:
            epochs.append(len(df_lc))
        else:
            best = float("inf")
            no_improved = 0
            for epoch, row in df_lc.iterrows():
                if row["valid"] < best:
                    best = row["valid"]
                    no_improved = 0
                else:
                    no_improved += 1
                
                if no_improved >= params_train.es_count:
                    break
            epochs.append(epoch + 1 - no_improved)
    epoch_avg = sum(epochs) // len(epochs)
    epoch_avg = max(epoch_avg, params_train.min_epoch)
    params_train.max_epoch = epoch_avg
    params_train.min_epoch = epoch_avg
    params_train.es_count = epoch_avg
    
    # prepare dataset
    dataset = params_data.g_data()
    print("len(dataset): ", len(dataset))

    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    dataset_full = Subset(dataset, indices)
    dataset_dummy = Subset(dataset, [0])

    loader_full = DataLoader(dataset_full, batch_size=params_train.batch_size, shuffle=True)
    loader_dummy = DataLoader(dataset_dummy, batch_size=params_train.batch_size, shuffle=True)

    # construct model
    params_model.output_allsite = False
    model = create_e3nn_model(e3nn_params=params_model)
    
    # prepare training
    dir_full = Path(dir_best) / "fulltrain"
    assert not dir_full.exists(), "Full training has already been done."
    os.makedirs(dir_full)
    params_train.path_checkpoint = dir_full / "checkpoint.pth.tar"
    params_train.path_modelbest = dir_full / "model_best.pth.tar"
    
    # train model
    train_e3nn_model(
        model=model,
        train_loader=loader_full,
        valid_loader=loader_dummy,
        params_data=params_data,
        params_model=params_model,
        params_train=params_train,
    )


if __name__ == "__main__":
    main()
