"""
Main script for panoptic experiments
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import argparse
import json
import os
import pprint

import numpy as np
import torch
import torch.utils.data as data

from train_panoptic import iterate, save_results
from src import model_utils
from src.dataset import PASTIS_Dataset
from src.panoptic.paps_loss import PaPsLoss
from src.utils import pad_collate

parser = argparse.ArgumentParser()
parser.add_argument(
    "--weight_folder",
    type=str,
    default="",
    help="Path to the main folder containing the pre-trained weights",
)
parser.add_argument(
    "--dataset_folder",
    default="",
    type=str,
    help="Path to the folder where the results are saved.",
)
parser.add_argument(
    "--res_dir",
    default="./inference_paps",
    type=str,
    help="Path to directory where results are written."
)
parser.add_argument(
    "--num_workers", default=8, type=int, help="Number of data loading workers"
)
parser.add_argument(
    "--fold",
    default=None,
    type=int,
    help="Do only one of the five fold (between 1 and 5)",
)
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="Name of device to use for tensor computations (cuda/cpu)",
)
parser.add_argument(
    "--display_step",
    default=50,
    type=int,
    help="Interval in batches between display of training metrics",
)


def main(config):
    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)

    device = torch.device(config.device)

    fold_sequence = [
        [[1, 2, 3], [4], [5]],
        [[2, 3, 4], [5], [1]],
        [[3, 4, 5], [1], [2]],
        [[4, 5, 1], [2], [3]],
        [[5, 1, 2], [3], [4]],
    ]

    fold_sequence = (
        fold_sequence if config.fold is None else [fold_sequence[config.fold - 1]]
    )

    model = model_utils.get_model(config, mode="panoptic")
    model = model.to(device)
    print(model)
    print("TOTAL TRAINABLE PARAMETERS :", config.N_params)

    for fold, (train_folds, val_fold, test_fold) in enumerate(fold_sequence):
        if config.fold is not None:
            fold = config.fold - 1
        dt_args = dict(
            folder=config.dataset_folder,
            norm=True,
            reference_date=config.ref_date,
            mono_date=config.mono_date,
            target="instance",
        )

        dt_test = PASTIS_Dataset(**dt_args, folds=test_fold)

        test_loader = data.DataLoader(
            dt_test,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=pad_collate,
            num_workers=config.num_workers,
        )

        # Load weights
        sd = torch.load(
            os.path.join(config.weight_folder, "Fold_{}".format(fold+1), "model.pth.tar"),
            map_location=device,
        )
        model.load_state_dict(sd["state_dict"])

        criterion = PaPsLoss(
            l_center=config.l_center,
            l_size=config.l_size,
            l_shape=config.l_shape,
            l_class=config.l_class,
            beta=config.beta,
            void_label=config.void_label,
        )
        print("Testing . . .")
        model.eval()
        test_metrics, tables = iterate(
            model,
            data_loader=test_loader,
            criterion=criterion,
            config=config,
            optimizer=None,
            mode="test",
            device=device,
        )
        save_results(fold + 1, test_metrics, tables, config)


if __name__ == "__main__":
    test_config = parser.parse_args()

    with open(os.path.join(test_config.weight_folder, "conf.json")) as file:
        model_config = json.loads(file.read())

    config = {**model_config, **vars(test_config)}
    config = argparse.Namespace(**config)
    config.fold = test_config.fold

    pprint.pprint(config)
    main(config)
