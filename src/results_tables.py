import os

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from .static.glioma_types import glioma_types
from .static.formatting import TABLE_FORMATTER
from config import SAVED_PREDICTIONS_DIR, RESULTS_DIR

models = ['base_model', 'convolutional_hlr_1', 'convolutional_hlr_5', 'fine_tune', 'MC_decoders', 'MC_center']
n_folds = 5


def get_gt_for_dataset(dataset):
    all_gt = []
    for fold in range(n_folds):
        all_gt.append(get_gt_for_dataset_and_fold(dataset, fold))

    return np.vstack(all_gt)


def get_gt_for_dataset_and_fold(dataset='BRATS_2018', fold=0):
    return np.load(os.path.join(SAVED_PREDICTIONS_DIR, dataset, "GT", "validation", f"GT_{fold}.npy"))


def get_volumes(dataset):
    gt = get_gt_for_dataset(dataset)
    volumes = np.sum(gt, axis=(1, 2, 3, 4))
    return volumes


def format_table_correlation(result_path, datasets, losses, cond='per_volume', correlation='pearson'):
    volumes_brats = get_volumes('BRATS_2018')
    volumes_isles = get_volumes('ISLES_2018')

    VOLUMES = {
        'BRATS_2018': volumes_brats,
        'ISLES_2018': volumes_isles
    }

    result_file = open(os.path.join(RESULTS_DIR, f'correlation_table_{correlation}_{cond}.txt'), 'w')
    print(f"Saving file at {result_file.name}")
    final_string = ""
    all_corrs_brats = []
    all_corrs_isles = []
    for model in models:
        row_string = TABLE_FORMATTER[model]
        for dataset in datasets:

            dataset_path = os.path.join(result_path, dataset)
            for loss in losses:
                path_to_files = os.path.join(dataset_path, 'all_points')
                per_volume_bias = np.load(path_to_files + f'/{loss}_{model}_{cond}_bias.npy')
                volumes = VOLUMES[dataset]

                if correlation == 'pearson':
                    corr, _ = pearsonr(per_volume_bias, volumes)
                    # Bowley 1928
                    se = (1 - corr**2) / np.sqrt(len(volumes))
                elif correlation == 'spearman':
                    corr, _ = spearmanr(per_volume_bias, volumes)
                    if dataset == 'BRATS_2018':
                        all_corrs_brats.append(corr)
                    else:
                        all_corrs_isles.append(corr)
                    # Bonnett and Wright (2000)
                    se = np.sqrt((1 + (corr**2 / 2)) / (len(volumes) - 3))
                elif correlation == 'kendall':
                    corr, _ = kendalltau(per_volume_bias, volumes)
                    if dataset == 'BRATS_2018':
                        all_corrs_brats.append(corr)
                    else:
                        all_corrs_isles.append(corr)
                    # Bonnett and Wright (2000)
                    se = np.sqrt(0.437 / (len(volumes) - 4))

                row_string += f" & {corr:.2f}  $\\pm$ {se:.2f}"

        final_string += row_string + " \\\\ \n"

    print(f"Len of all_corrs_brats: {len(all_corrs_brats)}")
    print(f"Len of all_corrs_isles: {len(all_corrs_isles)}")
    print(f"Median for Brats and metric {correlation} is {np.median(all_corrs_brats)}")
    print(f"Median for Isles and metric {correlation} is {np.median(all_corrs_isles)}")

    print(final_string)
    print("Saving...")
    result_file.write(final_string)


def format_table_hgg_vs_lgg(result_path, datasets, losses, cond='per_volume'):
    for gtype_idx, gtype in enumerate(['hgg', 'lgg']):
        final_string = ""
        result_file = open(os.path.join(result_path, f'{gtype}_bias_ece_table_{cond}.txt'), 'w')
        dataset_path = os.path.join(result_path, datasets[0])
        path_to_files = os.path.join(dataset_path, 'all_points')
        glioma_types_np = np.asarray(glioma_types, dtype=np.int)
        for model in models:
            row_string = TABLE_FORMATTER[model] + ' ;'
            for col in ['bias', 'ece']:
                for loss in losses:
                    if col == 'bias':
                        per_volume_bias = np.load(path_to_files + f'/{loss}_{model}_{cond}_bias.npy')
                        metric = np.abs(per_volume_bias)
                    else:
                        metric = np.load(path_to_files + f'/{loss}_{model}_{cond}_ece.npy')

                    metric_per_glioma_type = metric[np.where(glioma_types_np == gtype_idx)]
                    se = np.std(metric_per_glioma_type) / np.sqrt(len(metric_per_glioma_type))

                    row_string += f" {np.mean(metric_per_glioma_type):.4f} ± {se:.4f} ;"

            final_string += row_string + " \n"

        print(final_string)
        print(f"Saving for glioma type {gtype}...")
        result_file.write(final_string)


def format_table_bias_ece(result_path, datasets, losses, cond='per_volume'):
    for dataset in datasets:
        final_string = ""
        result_file = open(os.path.join(result_path, f'{dataset}_bias_ece_table_{cond}.txt'), 'w')
        dataset_path = os.path.join(result_path, dataset)
        path_to_files = os.path.join(dataset_path, 'all_points')
        for model in models:
            row_string = TABLE_FORMATTER[model]
            for col in ['bias', 'ece']:
                for loss in losses:
                    if col == 'bias':
                        per_volume_bias = np.load(path_to_files + f'/{loss}_{model}_{cond}_bias.npy')
                        metric = np.abs(per_volume_bias)
                    else:
                        metric = np.load(path_to_files + f'/{loss}_{model}_{cond}_ece.npy')

                    se = np.std(metric) / np.sqrt(len(metric))
                    row_string += f" & {np.mean(metric):.4f}  $\\pm$ {se:.4f}"

            final_string += row_string + " \\\\ \n"

        print(final_string)
        print("Saving...")
        result_file.write(final_string)
