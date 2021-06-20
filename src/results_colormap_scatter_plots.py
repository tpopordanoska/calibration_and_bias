import os
from datetime import datetime

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from src.static.formatting import LOSS_FORMATTER, DATASET_FORMATTER
from src.common.utils import create_folder, get_gt_for_dataset_and_fold

sns.set()
sns.set_context("paper")
sns.set_palette("pastel")

n_folds = 5
models = {
    'base_model': 'base model',
    'convolutional_hlr_1': 'Platt',
    'convolutional_hlr_5': 'auxiliary',
    'fine_tune': 'fine-tune',
    'MC_decoders': 'MC-Decoder',
    'MC_center': 'MC-Center'
}


FONTSIZE = 18
plt.rc('xtick', labelsize=FONTSIZE)
plt.rc('ytick', labelsize=FONTSIZE)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def get_volumes(dataset):
    gt = get_gt_for_dataset(dataset)
    volumes = np.sum(gt, axis=(1, 2, 3, 4))
    return volumes


def get_gt_for_dataset(dataset):
    all_gt = []
    for fold in range(n_folds):
        all_gt.append(get_gt_for_dataset_and_fold(dataset, fold))

    return np.vstack(all_gt)


def get_axislim(dataset):
    if "BRATS" in dataset:
        return (0, 0.15), (-0.15, 0.15)
    return (0, 0.1), (-0.1, 0.1)


def plot_colormap_scatter(result_path, datasets, losses, cond='per_volume'):
    volumes_brats = get_volumes('BRATS_2018') * 0.008
    volumes_isles = get_volumes('ISLES_2018') * 0.008
    all_volumes = np.concatenate([volumes_brats, volumes_isles], axis=0)
    min_, max_ = all_volumes.min(), all_volumes.max()

    plt.figure()
    all_volumes = []
    fig, axs = plt.subplots(nrows=2*len(models), ncols=3, figsize=(12, 20))
    my_cmap = truncate_colormap(sns.color_palette("magma_r", as_cmap=True), 0.2, 1)  # crest, rocket
    i = 0
    for dataset_i, dataset in enumerate(datasets):
        volumes = get_volumes(dataset) * 0.008
        all_volumes.extend(volumes)
        dataset_path = os.path.join(result_path, dataset)
        path = create_folder(result_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S ') + cond)
        for model_i, key in enumerate(models.keys()):
            for loss_i, loss in enumerate(losses):
                path_to_files = os.path.join(dataset_path, 'all_points')
                per_volume_bias = np.load(path_to_files + f'/{loss}_{key}_{cond}_bias.npy')
                abs_per_volume_bias = np.abs(per_volume_bias)

                print(len(per_volume_bias))
                per_volume_ece = np.load(path_to_files + f'/{loss}_{key}_{cond}_ece.npy')
                x = np.linspace(0, 0.15, 20)
                y_pos = np.abs(x)
                y_neg = -np.abs(x)

                # Calculate the correlation with the absolute value
                pearson_corr, _ = pearsonr(per_volume_ece, abs_per_volume_bias)
                spearman_corr, _ = spearmanr(per_volume_ece, abs_per_volume_bias)

                # Plot without the absolute value
                axs[i, loss_i].plot_separate_images(x, y_pos, zorder=0, c='grey')
                axs[model_i + 2 * dataset_i, loss_i].plot_separate_images(x, y_neg, zorder=0, c='grey')
                # s=300*volumes/max(volumes)
                pcm = axs[i, loss_i].scatter(x=per_volume_ece, y=per_volume_bias,
                                             c=volumes, cmap=my_cmap, s=100, zorder=10, vmin=min_, vmax=max_)

                if i == 0:
                    axs[i, loss_i].set_title(f"{LOSS_FORMATTER[loss]} \n "
                                             f"\n Pearson corr.: {pearson_corr:.2} ", fontsize=FONTSIZE)
                else:
                    axs[i, loss_i].set_title(f"Pearson corr.: {pearson_corr:.2}", fontsize=FONTSIZE)

                axs[i, loss_i].set_ylabel(f' {DATASET_FORMATTER[dataset]} '
                                          f'\n {models[key]} \n \n Bias', fontsize=FONTSIZE)

            i += 1

        for ax in axs.flat:
            ax.set_xlabel('ECE', fontsize=FONTSIZE)
            ax.set_xlim(0, 0.15)
            ax.set_ylim(-0.15, 0.15)
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

    fig.tight_layout(pad=2.0)
    col = fig.colorbar(pcm, ax=axs.flat, shrink=0.6, location='bottom', pad=0.1)
    col.set_label('ml', fontsize=FONTSIZE)
    fig.subplots_adjust(bottom=0.22)  # 0.21
    print(f"Saving at {path}\\_{cond}.pdf")
    plt.show()
    fig.savefig(f"{path}\\all_{cond}.pdf", bbox_inches='tight')
