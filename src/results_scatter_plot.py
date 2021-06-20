import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from .static.formatting import *

sns.set()
sns.set(font_scale=1)
sns.set_style("darkgrid", {"axes.facecolor": ".9"})


def plot_scatter_all_methods(result_path, datasets, cond="per_volume"):
    df1 = collect_results(os.path.join(result_path, datasets[0]), f'results_{cond}.csv')
    df2 = collect_results(os.path.join(result_path, datasets[1]), f'results_{cond}.csv')
    master_df = pd.concat([df1.assign(dataset="BRATS"), df2.assign(dataset="ISLES")])

    g = sns.FacetGrid(master_df, col="dataset", sharex=False, sharey=False)
    g.map_dataframe(sns.scatterplot, x="mean_ece", y="mean_abs_bias", hue="Loss", style="Method",  s=120)
    g.set_axis_labels("ECE", "$|\\operatorname{Bias}|$", fontsize=12)
    g.add_legend(markerscale=1.5, fontsize=10, bbox_to_anchor=(1, 0.5))

    axes = g.axes.flatten()
    axes[0].set_title("BR18")
    axes[1].set_title("IS18")
    plt.subplots_adjust(bottom=0.16)

    pearson_corr_1, _ = pearsonr(df1["mean_ece"], df1["mean_abs_bias"])
    pearson_se_1 = (1 - pearson_corr_1**2) / np.sqrt(len(df1["mean_ece"]))
    print(f"Pearson corr: {pearson_corr_1:.2f} +- {pearson_se_1:.2f}")

    pearson_corr_2, _ = pearsonr(df2["mean_ece"], df2["mean_abs_bias"])
    pearson_se_2 = (1 - pearson_corr_2**2) / np.sqrt(len(df2["mean_ece"]))
    print(f"Pearson corr: {pearson_corr_2:.2f} +- {pearson_se_2:.2f}")

    print(f"Saving at {result_path}\\grouped_{cond}_all_methods.pdf")
    plt.savefig(f"{result_path}\\grouped_{cond}_all_methods.png", bbox_inches='tight', transparent=True)


def collect_results(path, filename):
    print(path)
    results_CE = pd.read_csv(os.path.join(path, 'CE', filename))
    results_CESD = pd.read_csv(os.path.join(path, 'CE-SD', filename))
    results_SD = pd.read_csv(os.path.join(path, 'SD', filename))

    results_CE['loss'] = ["CE"] * len(results_CE.method)
    results_CESD['loss'] = ["CE-SD"] * len(results_CESD.method)
    results_SD['loss'] = ["SD"] * len(results_SD.method)

    frames = [results_CE, results_CESD, results_SD]
    result = pd.concat(frames)

    for method_key, method_value in MODEL_FORMATTER.items():
        result.replace(method_key, method_value, inplace=True)

    for loss_key, loss_value in LOSS_FORMATTER.items():
        result.replace(loss_key, loss_value, inplace=True)

    # Get bias in ml
    result.mean_abs_bias = result.mean_abs_bias * 0.008
    result.rename(columns={"method": "Method", "loss": "Loss"}, inplace=True)

    return result


def correlation_lgg_hgg(ece_bias_hgg_lgg_path, dataset, cond='per_volume'):
    df_hgg = collect_results(os.path.join(ece_bias_hgg_lgg_path, dataset), f'results_hgg_{cond}.csv')
    df_lgg = collect_results(os.path.join(ece_bias_hgg_lgg_path, dataset), f'results_lgg_{cond}.csv')

    pearson_corr_hgg, _ = pearsonr(df_hgg["mean_ece"], df_hgg["mean_abs_bias"])
    pearson_se_hgg = (1 - pearson_corr_hgg**2) / np.sqrt(len(df_hgg["mean_ece"]))
    print(f"Pearson corr for HGG: {pearson_corr_hgg:.2f} +- {pearson_se_hgg:.2f}")

    pearson_corr_lgg, _ = pearsonr(df_lgg["mean_ece"], df_lgg["mean_abs_bias"])
    pearson_se_lgg = (1 - pearson_corr_lgg**2) / np.sqrt(len(df_lgg["mean_ece"]))
    print(f"Pearson corr for LGG: {pearson_corr_lgg:.2f} +- {pearson_se_lgg:.2f}")
