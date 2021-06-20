import pandas as pd

from config import ROOT_DIR
from .common import metrics
from .common.utils import *
from .static.glioma_types import glioma_types

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

glioma_types_np = np.asarray(glioma_types, dtype=np.int)


def save_bias_and_ece(kernel_sizes, nr_bins, result_path, datasets, cond='per_volume', tumor_splits=3):
    for dataset in datasets:
        save_dir = create_nested_folders(result_path, [dataset, 'all_points'])
        results_dict = dict()  # for the scatter plot
        point_dir = os.path.join(SAVED_PREDICTIONS_DIR, dataset, "Checkpoints")
        for loss in os.listdir(point_dir):
            result_list = []
            for method in os.listdir(os.path.join(point_dir, loss)):
                for k in kernel_sizes:
                    all_preds, all_gt, all_masks = read_data(dataset, loss, method, k, point_dir)
                    method_name = get_formatted_method_name(method, k)
                    name = f'{loss}_{method_name}_{cond}'
                    ece, bias = [-1], [-1]

                    if cond == 'per_volume':
                        ece, bias = get_and_save_bias_and_ece(all_gt, all_preds, all_masks, nr_bins, save_dir, name)
                    elif 'per_tumor_size' in cond:
                        which_tumor_split = cond.split("_")[-1]
                        binids = split_by_tumor_size(all_gt, tumor_splits)
                        print(np.bincount(binids))
                        idx = np.where(binids == int(which_tumor_split))
                        preds, gt, masks = all_preds[idx], all_gt[idx], all_masks[idx]
                        ece, bias = get_and_save_bias_and_ece(gt, preds, masks, nr_bins, save_dir, name)

                    mean_ece = ece.mean()
                    mean_abs_bias = np.abs(bias).mean()
                    result_list.append([method_name, mean_ece, mean_abs_bias])
                    results_dict[f"{loss}_{method_name}"] = (mean_ece, mean_abs_bias)

                    if 'convolutional' not in method:
                        break

            save_result_list(result_list, result_path, dataset, loss, f"{cond}")


def save_bias_and_ece_per_glioma_type(kernel_sizes, nr_bins, result_path, cond='per_volume'):
    dataset = 'BRATS_2018'
    results_dict = dict()  # for the scatter plot
    point_dir = os.path.join(SAVED_PREDICTIONS_DIR, dataset, "Checkpoints")
    for gtype_idx, gtype in enumerate(['hgg', 'lgg']):
        for loss in os.listdir(point_dir):
            result_list = []
            for method in os.listdir(os.path.join(point_dir, loss)):
                for k in kernel_sizes:
                    all_preds, all_gt, all_masks = read_data(dataset, loss, method, k, point_dir)
                    method_name = get_formatted_method_name(method, k)

                    gtype_idx_to_select = np.where(glioma_types_np == gtype_idx)
                    gt_per_gtype = all_gt[gtype_idx_to_select]
                    preds_per_gtype = all_preds[gtype_idx_to_select]
                    masks_per_gtype = all_masks[gtype_idx_to_select]

                    ece = metrics.per_volume_masked_ECE(gt_per_gtype, preds_per_gtype, masks_per_gtype, n_bins=nr_bins)
                    bias = metrics.per_volume_masked_bias(gt_per_gtype, preds_per_gtype, masks_per_gtype)
                    mean_ece = ece.mean()
                    mean_abs_bias = np.abs(bias).mean()

                    result_list.append([method_name, mean_ece, mean_abs_bias])
                    results_dict[f"{loss}_{method_name}"] = (mean_ece, mean_abs_bias)

                    if 'convolutional' not in method:
                        break

            save_result_list(result_list, result_path, dataset, loss, f"{gtype}_{cond}")


def save_result_list(result_list, result_path, dataset, loss, name):
    df = pd.DataFrame(result_list, columns=['method', 'mean_ece', 'mean_abs_bias'])
    df = df.round({'mean_ece': 4, 'mean_abs_bias': 4})
    df.set_index(['method'])
    path = os.path.join(ROOT_DIR, result_path, dataset, loss)
    print(f"Saving results to {os.path.join(path, f'results_{name}.csv')}")
    if not os.path.exists(path):
        os.makedirs(path)

    df.to_csv(os.path.join(path, f'results_{name}.csv'))


def get_and_save_bias_and_ece(gt, preds, masks, nr_bins, save_dir, save_name):
    ece = metrics.per_volume_masked_ECE(gt, preds, masks, n_bins=nr_bins)
    bias = metrics.per_volume_masked_bias(gt, preds, masks)

    np.save(os.path.join(save_dir, f'{save_name}_ece.npy'), ece)
    np.save(os.path.join(save_dir, f'{save_name}_bias.npy'), bias)

    return ece, bias


def read_data(dataset, loss, method, k, point_dir, n_folds=5):
    print(f"Calculating metrics for {dataset, loss, method}")
    # Initialize/clear results
    all_preds, all_gt, all_masks = [], [], []

    for fold in range(n_folds):
        # Load data
        preds = get_predictions_for_fold(point_dir, loss, method, fold, k)
        gt = get_gt_for_dataset_and_fold(dataset, fold)
        masks = get_masks_for_dataset_and_fold(dataset, fold)

        if preds.shape[-1] != 1 and gt.shape[-1] == 1:
            preds = preds[..., None]
        assert(gt.shape == masks.shape)

        p_max = preds.max()
        if p_max > 1.1:
            # preds are logits
            preds = sigmoid(preds)
        # Collect predictions for all 5 folds
        all_preds.append(preds)
        all_gt.append(gt)
        all_masks.append(masks)
    all_preds = np.vstack(all_preds)
    all_gt = np.vstack(all_gt)
    all_masks = np.vstack(all_masks)

    return all_preds, all_gt, all_masks


def get_formatted_method_name(method, k):
    if 'convolutional' in method:
        return f"{method}_{k}"
    if 'MC_center' in method:
        return 'MC_center'
    if 'MC_decoders' in method:
        return "MC_decoders"

    return method


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def split_by_tumor_size(gt, tumor_splits):
    volumes = np.sum(gt, axis=(1, 2, 3, 4))
    bins = np.linspace(min(volumes),  max(volumes) - max(volumes)/tumor_splits, tumor_splits)
    return np.digitize(volumes, bins) - 1
