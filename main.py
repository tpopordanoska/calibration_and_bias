from config import RESULTS_DIR
from src.common.utils import create_nested_folders
from src.prepare_bias_and_ece import save_bias_and_ece, save_bias_and_ece_per_glioma_type
from src.results_scatter_plot import plot_scatter_all_methods, correlation_lgg_hgg
from src.results_colormap_scatter_plots import plot_colormap_scatter
from src.results_separate_images import save_images_for_plotting, plot_separate_images
from src.results_tables import format_table_bias_ece, format_table_correlation, format_table_hgg_vs_lgg

n_folds = 5
tumor_splits = 3
nr_bins = 20
kernel_sizes = [1, 5]
losses = ["CE", "CE-SD", "SD"]
datasets = ['BRATS_2018', "ISLES_2018"]

if __name__ == '__main__':

    result_path = create_nested_folders(RESULTS_DIR, ['ece_vs_bias'])
    save_bias_and_ece(kernel_sizes, nr_bins, result_path, datasets)

    result_path_hgg_lgg = create_nested_folders(RESULTS_DIR, ['ece_vs_bias_hgg_lgg'])
    save_bias_and_ece_per_glioma_type(kernel_sizes, nr_bins, result_path_hgg_lgg)

    plot_scatter_all_methods(result_path, datasets)
    correlation_lgg_hgg(result_path_hgg_lgg, datasets[0])

    plot_colormap_scatter(result_path, datasets, losses)

    save_images_for_plotting()
    plot_separate_images(result_path, datasets)

    format_table_bias_ece(result_path, datasets, losses)
    format_table_correlation(result_path, datasets, losses, 'per_volume', 'spearman')
    format_table_hgg_vs_lgg(result_path, datasets, losses, 'per_volume')
