import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve

from config import ROOT_DIR, SEP_IMAGE_PATH, RESULTS_DIR
from .common.utils import *
from .static.formatting import DATASET_FORMATTER, MODEL_FORMATTER

n_folds = 5

sns.set()
sns.set_context("paper")
sns.set_palette("deep")

FONTSIZE = 16
plt.rc('xtick', labelsize=FONTSIZE)
plt.rc('ytick', labelsize=FONTSIZE)

titles = [
    'BRATS base model', 'BRATS fine-tune', 'ISLES base model', 'ISLES fine-tune'
]


# SAVE IMAGES FOR PLOTTING
def save_images_for_plotting():
    for dataset in ['BRATS_2018', 'ISLES_2018']:
        point_dir = os.path.join(SAVED_PREDICTIONS_DIR, dataset, "Checkpoints")
        gt = get_gt_for_dataset(dataset)
        original = get_original_for_dataset(dataset)

        sep_images_path = create_folder(os.path.join(ROOT_DIR, RESULTS_DIR), "separate_images")
        im1_idx, im2_idx = choose_images(gt)
        for idx, model in zip([im1_idx, im2_idx], ['base_model', 'fine_tune']):
            predictions_CE = get_predictions(point_dir, loss="CE", method=model)
            predictions_CESD = get_predictions(point_dir, loss="CE-SD", method=model)
            predictions_SD = get_predictions(point_dir, loss="SD", method=model)

            np.save(os.path.join(sep_images_path, f"{dataset}_{model}_gt_{idx}"), gt[idx])
            np.save(os.path.join(sep_images_path, f"{dataset}_{model}_orig_{idx}"), original[idx])
            np.save(os.path.join(sep_images_path, f"{dataset}_{model}_pred_CE_{idx}"), predictions_CE[idx])
            np.save(os.path.join(sep_images_path, f"{dataset}_{model}_pred_CESD_{idx}"), predictions_CESD[idx])
            np.save(os.path.join(sep_images_path, f"{dataset}_{model}_pred_SD_{idx}"), predictions_SD[idx])


def get_predictions(point_dir, loss, method):
    all_preds = []
    for fold in range(n_folds):
        all_preds.append(get_predictions_for_fold(point_dir, loss, method, fold))

    return np.vstack(all_preds)


def get_gt_for_dataset(dataset):
    all_gt = []
    for fold in range(n_folds):
        all_gt.append(get_gt_for_dataset_and_fold(dataset, fold))

    return np.vstack(all_gt)


def get_original_for_dataset(dataset):
    all_original = []
    for fold in range(n_folds):
        all_original.append(get_orig_preprocessed_for_dataset_and_fold(dataset, fold))

    return np.vstack(all_original)


def get_orig_preprocessed_for_dataset_and_fold(dataset='BRATS_2018', fold=0):
    return np.load(os.path.join(SAVED_PREDICTIONS_DIR, dataset, "original", "validation", f"X_{fold}.npy"))


def choose_images(gt, perc1=0.4, perc2=0.9):
    volumes = np.sum(gt, axis=(1, 2, 3, 4))
    volumes_sorted = sorted(volumes)
    val1, val2 = get_index_at_percentile(volumes_sorted, perc1), get_index_at_percentile(volumes_sorted, perc2)
    idx1, idx2 = np.where(volumes == val1), np.where(volumes == val2)
    return idx1[0][0], idx2[0][0]


def get_index_at_percentile(a_list, p):
    n = max(int(round(p * len(a_list) + 0.5)), 2)
    return a_list[n]


# PLOT THE SAVED IMAGES
def plot_separate_images(results_path, datasets):
    plt.figure()
    selected_img_idx_brats = [153, 58]
    selected_img_idx_isles = [49, 68]
    idx_total = len(selected_img_idx_brats) + len(selected_img_idx_isles)

    n_columns = 4
    plt.figure(figsize=(idx_total * 3, n_columns * 3))
    gs1 = gridspec.GridSpec(idx_total, n_columns)
    gs1.update(wspace=0.2, hspace=0.01)
    i = 0
    cc_gt, cc_pred_ce, cc_pred_cesd, cc_pred_sd = [], [], [], []
    for dataset in datasets:
        axes = []
        if dataset == 'BRATS_2018':
            selected_idx = selected_img_idx_brats
            channel = 0
        else:
            selected_idx = selected_img_idx_isles
            channel = 1

        for idx in selected_idx:
            if i % 2 == 0:
                model = 'base_model'
            else:
                model = 'fine_tune'
            gt, orig, pred_CE, pred_CESD, pred_SD = load_images_for_plotting(RESULTS_DIR, dataset, model, idx)
            pred_CE = get_sigmoid_values(pred_CE)
            pred_CESD = get_sigmoid_values(pred_CESD)
            pred_SD = get_sigmoid_values(pred_SD)

            print("Prediction ce after sigmoid", pred_SD.shape)
            bias_CE, bias_CESD, bias_SD, ece_CE, ece_CESD, ece_SD = load_ece_and_bias(results_path, dataset, model, idx)
            cc_gt.append(gt)
            cc_pred_ce.append(pred_CE)
            cc_pred_cesd.append(pred_CESD)
            cc_pred_sd.append(pred_SD)

            selected_slice = pred_SD.shape[2] // 2

            selected_gt = gt[:, :, selected_slice, 0]
            gt_boundary = selected_gt.copy()
            gt_boundary[gt_boundary == 1] = 0
            gt_boundary[gt_boundary > 0] = 1

            selected_orig = orig[:, :, selected_slice, channel]
            selected_pred_CE = pred_CE[:, :, selected_slice, 0]
            selected_pred_CESD = pred_CESD[:, :, selected_slice, 0]
            selected_pred_SD = pred_SD[:, :, selected_slice, 0]

            save_pngs(dataset, idx, selected_gt, selected_orig, selected_pred_CE, selected_pred_CESD, selected_pred_SD)
            mask, orig, pred_CE, pred_CESD, pred_SD, pred_CE_t, pred_CESD_t, pred_SD_t = load_pngs(dataset, idx)

            orig = overlay_mask(orig, mask, line_color=(255, 0, 0), crop=True)
            pred_CE = overlay_mask(pred_CE, mask, line_color=(255, 0, 0))
            pred_CESD = overlay_mask(pred_CESD, mask, line_color=(255, 0, 0))
            pred_SD = overlay_mask(pred_SD, mask, line_color=(255, 0, 0))

            pred_CE = overlay_mask(pred_CE, pred_CE_t, line_color=(0, 0, 255), crop=True)
            pred_CESD = overlay_mask(pred_CESD, pred_CESD_t, line_color=(0, 0, 255), crop=True)
            pred_SD = overlay_mask(pred_SD, pred_SD_t, line_color=(0, 0, 255), crop=True)

            n = np.arange(n_columns)

            ax1 = plt.subplot(gs1[i, n[0]])
            ax1.imshow(orig, cmap='Greys')
            if i == 0:
                ax1.set_title("Original", fontsize=16)
            ax1.set_ylabel(f"{DATASET_FORMATTER[dataset]} \n {MODEL_FORMATTER[model]}", fontsize=12)

            ax2 = plt.subplot(gs1[i, n[1]])
            ax2.imshow(pred_CE, cmap='Greys')
            if i == 0:
                ax2.set_title(f"CrE", fontsize=16)
            else:
                ax2.set_title("")
            ax2.set_xlabel(f"ECE: {ece_CE:.3f}, Bias: {bias_CE:.3f}", fontsize=13)

            ax3 = plt.subplot(gs1[i, n[2]])
            ax3.imshow(pred_CESD, cmap='Greys')
            if i == 0:
                ax3.set_title(f"CrE-SD", fontsize=16)
            else:
                ax3.set_title("")
            ax3.set_xlabel(f"ECE: {ece_CESD:.3f}, Bias: {bias_CESD:.3f}", fontsize=13)

            ax4 = plt.subplot(gs1[i, n[3]])
            ax4.imshow(pred_SD, cmap='Greys')
            if i == 0:
                ax4.set_title(f"SD", fontsize=16)
            else:
                ax4.set_title(f"")
            ax4.set_xlabel(f"ECE: {ece_SD:.3f}, Bias: {bias_SD:.3f}", fontsize=13)

            axes.append(ax1)
            axes.append(ax2)
            axes.append(ax3)
            axes.append(ax4)

            i += 1

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

    plt.savefig(f"{RESULTS_DIR}\\examples.pdf", bbox_inches='tight')
    plt.show()
    plot_calibration_curve(cc_gt, cc_pred_ce, cc_pred_cesd, cc_pred_sd, RESULTS_DIR)


def save_as_png(path, img):
    plt.figure()
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.imshow(img,  cmap=plt.cm.gray)
    plt.savefig(path, bbox_inches='tight')


def load_images_for_plotting(result_path, dataset, model, idx):
    sep_images_path = os.path.join(ROOT_DIR, result_path, "separate_images")
    gt = np.load(os.path.join(sep_images_path, f"{dataset}_{model}_gt_{idx}.npy"))
    orig = np.load(os.path.join(sep_images_path, f"{dataset}_{model}_orig_{idx}.npy"))
    predCE = np.load(os.path.join(sep_images_path, f"{dataset}_{model}_pred_CE_{idx}.npy"))
    pred_CESD = np.load(os.path.join(sep_images_path, f"{dataset}_{model}_pred_CESD_{idx}.npy"))
    predSD = np.load(os.path.join(sep_images_path, f"{dataset}_{model}_pred_SD_{idx}.npy"))

    return gt, orig, predCE, pred_CESD, predSD


def threshold(predictions_CE, predictions_CESD, predictions_SD):
    predictions_CE[predictions_CE > 0.5] = 1
    predictions_CE[predictions_CE != 1] = 0
    predictions_CESD[predictions_CESD > 0.5] = 1
    predictions_CESD[predictions_CESD != 1] = 0
    predictions_SD[predictions_SD > 0.5] = 1
    predictions_SD[predictions_SD != 1] = 0

    return predictions_CE, predictions_CESD, predictions_SD


def save_pngs(dataset, idx, selected_gt, orig, pred_CE, pred_CESD, pred_SD):
    if not os.path.exists(os.path.join(SEP_IMAGE_PATH, f"{dataset}_gt_{idx}.png")):
        save_as_png(os.path.join(SEP_IMAGE_PATH, f"{dataset}_gt_{idx}"), selected_gt)
    if not os.path.exists(os.path.join(SEP_IMAGE_PATH, f"{dataset}_orig_{idx}.png")):
        save_as_png(os.path.join(SEP_IMAGE_PATH, f"{dataset}_orig_{idx}"), orig)
    if not os.path.exists(os.path.join(SEP_IMAGE_PATH, f"{dataset}_CE_{idx}.png")):
        save_as_png(os.path.join(SEP_IMAGE_PATH, f"{dataset}_CE_{idx}"), pred_CE)
    if not os.path.exists(os.path.join(SEP_IMAGE_PATH, f"{dataset}_CESD_{idx}.png")):
        save_as_png(os.path.join(SEP_IMAGE_PATH, f"{dataset}_CESD_{idx}"), pred_CESD)
    if not os.path.exists(os.path.join(SEP_IMAGE_PATH, f"{dataset}_SD_{idx}.png")):
        save_as_png(os.path.join(SEP_IMAGE_PATH, f"{dataset}_SD_{idx}"), pred_SD)

    pred_CE_thresh, pred_CESD_thresh, pred_SD_thresh = threshold(pred_CE, pred_CESD, pred_SD)

    if not os.path.exists(os.path.join(SEP_IMAGE_PATH, f"{dataset}_CE_thresh_{idx}.png")):
        save_as_png(os.path.join(SEP_IMAGE_PATH, f"{dataset}_CE_thresh_{idx}"), pred_CE_thresh)
    if not os.path.exists(os.path.join(SEP_IMAGE_PATH, f"{dataset}_CESD_thresh_{idx}.png")):
        save_as_png(os.path.join(SEP_IMAGE_PATH, f"{dataset}_CESD_thresh_{idx}"), pred_CESD_thresh)
    if not os.path.exists(os.path.join(SEP_IMAGE_PATH, f"{dataset}_SD_thresh_{idx}.png")):
        save_as_png(os.path.join(SEP_IMAGE_PATH, f"{dataset}_SD_thresh_{idx}"), pred_SD_thresh)


def load_pngs(dataset, idx):
    orig = cv2.imread(os.path.join(SEP_IMAGE_PATH, f"{dataset}_orig_{idx}.png"))

    y_path = os.path.join(SEP_IMAGE_PATH, f"{dataset}_gt_{idx}.png")
    mask = ((cv2.cvtColor(cv2.imread(y_path), cv2.COLOR_BGR2GRAY) > 127) * 255).astype('uint8')

    pred_CE = cv2.imread(os.path.join(SEP_IMAGE_PATH, f"{dataset}_CE_{idx}.png"))
    pred_CESD = cv2.imread(os.path.join(SEP_IMAGE_PATH, f"{dataset}_CESD_{idx}.png"))
    pred_SD = cv2.imread(os.path.join(SEP_IMAGE_PATH, f"{dataset}_SD_{idx}.png"))

    ce_thresh_path = os.path.join(SEP_IMAGE_PATH, f"{dataset}_CE_thresh_{idx}.png")
    pred_CE_thresh = ((cv2.cvtColor(cv2.imread(ce_thresh_path), cv2.COLOR_BGR2GRAY) > 127) * 255).astype('uint8')

    cesd_tresh_path = os.path.join(SEP_IMAGE_PATH, f"{dataset}_CESD_thresh_{idx}.png")
    pred_CESD_thresh = ((cv2.cvtColor(cv2.imread(cesd_tresh_path), cv2.COLOR_BGR2GRAY) > 127) * 255).astype('uint8')

    sd_thresh_path = os.path.join(SEP_IMAGE_PATH, f"{dataset}_SD_thresh_{idx}.png")
    pred_SD_thresh = ((cv2.cvtColor(cv2.imread(sd_thresh_path), cv2.COLOR_BGR2GRAY) > 127) * 255).astype('uint8')

    return mask, orig, pred_CE, pred_CESD, pred_SD, pred_CE_thresh, pred_CESD_thresh, pred_SD_thresh


def overlay_mask(orig, mask, line_color=(0, 255, 0), fill_color=(0, 0, 0), transparency=0.5, crop=False):
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if fill_color == (0, 0, 0):
        transparency = 0
    color_mask = np.zeros(mask.shape + (3,))
    for i, c in enumerate(fill_color):
        color_mask[:, :, i] = c * (mask > 0)
    subtracted = np.clip(orig - transparency * cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0, 255)
    result = cv2.addWeighted(subtracted.astype('uint8'), 1, color_mask.astype('uint8'), transparency, 0)

    thresh, im_bw = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)  # im_bw: binary image
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(result, contours, -1, line_color, 3)

    if crop:
        margin = 20
        crop_img = result[margin:-margin, margin:-margin]
        result = cv2.rotate(crop_img, cv2.ROTATE_90_CLOCKWISE)

    return result


def load_ece_and_bias(results_path, dataset, model, idx):
    dataset_path = os.path.join(results_path, dataset)
    path_to_files = os.path.join(dataset_path, 'all_points')

    bias_CE = np.load(path_to_files + f'/CE_{model}_per_volume_bias.npy')[idx]
    bias_CESD = np.load(path_to_files + f'/CE-SD_{model}_per_volume_bias.npy')[idx]
    bias_SD = np.load(path_to_files + f'/SD_{model}_per_volume_bias.npy')[idx]

    ece_CE = np.load(path_to_files + f'/CE_{model}_per_volume_ece.npy')[idx]
    ece_CESD = np.load(path_to_files + f'/CE-SD_{model}_per_volume_ece.npy')[idx]
    ece_SD = np.load(path_to_files + f'/SD_{model}_per_volume_ece.npy')[idx]

    return bias_CE, bias_CESD, bias_SD, ece_CE, ece_CESD, ece_SD


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def get_sigmoid_values(preds):
    p_max = preds.max()
    if p_max > 1.1:
        # preds are logits
        preds = sigmoid(preds)

    return preds


def plot_calibration_curve(gts, pred_CEs, pred_CESDs, pred_SDs, result_path, n_bins=10):
    plt.figure()
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
    for i in range(len(gts)):
        gt, pred_CE, pred_CESD, pred_SD = gts[i], pred_CEs[i], pred_CESDs[i], pred_SDs[i]
        print("gt shape", gt.shape)
        print("predCE shape", pred_CE.shape)
        gt[gt > 0.5] = 1
        gt[gt != 1] = 0

        prob_true_ce, prob_pred_ce = calibration_curve(gt.flatten(), pred_CE.flatten(), n_bins=n_bins)
        prob_true_cesd, prob_pred_cesd = calibration_curve(gt.flatten(), pred_CESD.flatten(), n_bins=n_bins)
        prob_true_sd, prob_pred_sd = calibration_curve(gt.flatten(), pred_SD.flatten(), n_bins=n_bins)

        axs[i].plot(sorted(prob_true_ce), prob_pred_ce, "s-", label="CrE")
        axs[i].plot(sorted(prob_true_cesd), prob_pred_cesd, "s-", label="CrE-SD")
        axs[i].plot(sorted(prob_true_sd), prob_pred_sd, "s-", label="SD")
        axs[i].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        axs[i].set_ylabel("Fraction of positives", fontsize=FONTSIZE)
        axs[i].set_xlabel("Mean predicted value", fontsize=FONTSIZE)
        axs[i].set_title(titles[i], fontsize=FONTSIZE)

    for ax in axs.flat:
        ax.label_outer()

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    handles, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=FONTSIZE, markerscale=2)
    fig.subplots_adjust(bottom=0.3)
    plt.savefig(f"{result_path}\\cal_curves.pdf", bbox_inches='tight')
    plt.show()
