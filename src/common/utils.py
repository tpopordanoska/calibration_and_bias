import os

import numpy as np

from config import SAVED_PREDICTIONS_DIR


def create_nested_folders(base_path, folder_names):
    for folder_name in folder_names:
        base_path = create_folder(base_path, folder_name)

    return base_path


def create_folder(path, folder_name):
    path_folder = os.path.join(path, folder_name)
    if not os.path.exists(path_folder):
        try:
            os.mkdir(path_folder)
        except OSError:
            print("Creation of the directory %s failed" % path_folder)
    return path_folder


def get_gt_for_dataset_and_fold(dataset='BRATS_2018', fold=0):
    return np.load(os.path.join(SAVED_PREDICTIONS_DIR, dataset, "GT", "validation", f"GT_{fold}.npy"))


def get_masks_for_dataset_and_fold(dataset='BRATS_2018', fold=0):
    return np.load(os.path.join(SAVED_PREDICTIONS_DIR, dataset, "masks", "validation", f"mask_{fold}.npy"))


def get_predictions_for_fold(point_dir, loss, method, fold, k=1):
    if 'convolutional' in method:
        return np.load(os.path.join(point_dir, loss, method, "validation", f"predictions_{fold}_k_{k}.npy"))
    return np.load(os.path.join(point_dir, loss, method, "validation", f"predictions_{fold}.npy"))

