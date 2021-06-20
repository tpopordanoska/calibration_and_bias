import numpy as np


#  from: https://github.com/AxelJanRousseau/PostTrainCalibration/
def fast_ece(y_true, y_pred, n_bins=10):
    # ~sklearn code
    bins = np.linspace(0., 1. - 1./n_bins, n_bins)
    binids = np.digitize(y_pred, bins) - 1

    bin_sums = np.bincount(binids, weights=y_pred, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0  # don't use empty bins
    prob_true = (bin_true[nonzero] / bin_total[nonzero])  # acc
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])  # conf

    weights = bin_total[nonzero] / np.sum(bin_total[nonzero])
    l1 = np.abs(prob_true-prob_pred)
    ece = np.sum(weights*l1)
    mce = l1.max()
    l1 = l1.sum()

    return {"acc": prob_true, "conf": prob_pred, "ECE": ece, "MCE": mce, "l1": l1}


def per_volume_masked_ECE(y_true, y_pred, masks, n_bins=10):
    nr_volumes = len(y_true)
    result = np.zeros(nr_volumes)
    for i in range(nr_volumes):
        result[i] = fast_ece(y_true[i, masks[i]], y_pred[i, masks[i]], n_bins)["ECE"]

    return result


def per_dataset_maskedECE(y_true, y_pred, masks, n_bins=10):
    flat_y_true, flat_y_pred = [], []
    for i in range(len(y_true)):
        flat_y_true.extend(y_true[i, masks[i]])
        flat_y_pred.extend(y_pred[i, masks[i]])

    return fast_ece(flat_y_true, flat_y_pred, n_bins)["ECE"]


def binary_vol_diff(y_true, y_pred, threshold=None, per_image=True, voxel_volume=1):
    if threshold is not None:
        y_true = np.greater(y_true, threshold)
        y_pred = np.greater(y_pred, threshold)

    if per_image:
        return (np.sum(y_pred, axis=(1, 2, 3, 4)) - np.sum(y_true, axis=(1, 2, 3, 4))) * voxel_volume

    return [(np.sum(y_pred) - np.sum(y_true)) * voxel_volume]


def binary_rel_vol_diff(y_true, y_pred, threshold=None, per_image=True):
    vol_diff = binary_vol_diff(y_true, y_pred, threshold=threshold, per_image=per_image)
    return vol_diff / np.sum(y_true, axis=(1, 2, 3, 4) if per_image else None)


def per_volume_masked_bias(y_true, y_pred, masks):
    nr_volumes = len(y_true)
    biases = np.zeros(nr_volumes)
    for i in range(nr_volumes):
        biases[i] = np.mean(y_pred[i, masks[i]] - y_true[i, masks[i]])

    return biases


def per_volume_bias(y_true, y_pred):
    nr_volumes = len(y_true)
    biases = np.zeros(nr_volumes)
    for i in range(nr_volumes):
        biases[i] = np.mean(y_pred[i] - y_true[i])

    return biases


def per_dataset_bias(y_true, y_pred):
    return np.mean(y_pred - y_true)
