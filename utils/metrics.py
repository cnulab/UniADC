"""Anomaly metrics."""
import numpy as np
from sklearn import metrics
from numpy import ndarray
import pandas as pd
from skimage import measure
from sklearn.metrics import precision_recall_curve,average_precision_score,jaccard_score
import torch
import torch.nn.functional as F


def compute_imagewise_metrics(
    prediction_scores, gt_labels, **kwargs
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """

    auroc = metrics.roc_auc_score(
        gt_labels, prediction_scores
    )

    return {"image_auroc": auroc}


def compute_pixelwise_metrics(prediction_masks, gt_masks, **kwargs):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(prediction_masks, list):
        prediction_masks = np.stack(prediction_masks)
    if isinstance(gt_masks, list):
        gt_masks = np.stack(gt_masks)

    flat_anomaly_segmentations = prediction_masks.ravel()
    flat_ground_truth_masks = gt_masks.ravel()

    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    return {
        "pixel_auroc": auroc,
    }


def compute_iou_metrics(
    prediction_scores, gt_labels, num_classes, ignore_label = None,**kwargs
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    pred_flat = prediction_scores.ravel()
    mask_flat = gt_labels.ravel()
    if ignore_label is not None:
        pred_flat = pred_flat[mask_flat!=ignore_label]
        mask_flat = mask_flat[mask_flat!=ignore_label]
    iou = jaccard_score(mask_flat, pred_flat, average='macro', labels=list(range(num_classes)))
    return {"iou": iou}


def compute_pro(prediction_masks, gt_masks, num_th: int = 200, **kwargs):

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """
    assert isinstance(prediction_masks, ndarray), "type(amaps) must be ndarray"
    assert isinstance(gt_masks, ndarray), "type(masks) must be ndarray"
    assert prediction_masks.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert gt_masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert prediction_masks.shape == gt_masks.shape, "amaps.shape and masks.shape must be same"
    assert set(gt_masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(prediction_masks, dtype=bool)
    min_th = prediction_masks.min()
    max_th = prediction_masks.max()
    delta = (max_th - min_th) / num_th
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[prediction_masks <= th] = 0
        binary_amaps[prediction_masks > th] = 1
        pros = []
        for binary_amap, mask in zip(binary_amaps, gt_masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)
        inverse_masks = 1 - gt_masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        df_new = pd.DataFrame([{"pro": np.mean(pros), "fpr": fpr, "threshold": th}])
        df = pd.concat([df, df_new], ignore_index=True)
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()
    pro_auc = metrics.auc(df["fpr"], df["pro"])
    return {"pixel_pro":pro_auc}
