from typing import Dict, List

import numpy as np
import torch

from src.utils.misc import so3_rotation_angle


def mean_avg_precision_numpy(distances: np.ndarray, similarities: np.ndarray) -> float:
    """Compute the mean average precision, ie the average precision integrated over all distance thresholds, averaged over all query points.

    Parameters
    ----------
    distances (n_query x n_samples array):
        the 3D world distances between the (3D world points associated with the) query patch and the sample patches

    similarities (n_query x n_samples array):
        the similarity scores between the query patch and the sample patches

    Returns
    -------
    mean_avg_precision (float):
        the mean average precision, integrated over all distance thresholds, averaged over all query points
    """
    n_query, n_samples = distances.shape

    # sort by similarity first to implicitly use as tie-breaker
    # not the _most_ efficient, but this is the equivalent of what I have to do in PyTorch
    sort_idxs = np.argsort(similarities, axis=1)[:, ::-1]
    distances = np.take_along_axis(distances, sort_idxs, axis=1)
    similarities = np.take_along_axis(similarities, sort_idxs, axis=1)

    # sort distances increasing order
    sorted_dist_indices = np.argsort(distances, axis=1, kind="stable")
    distances = np.take_along_axis(distances, sorted_dist_indices, axis=1)
    similarities = np.take_along_axis(similarities, sorted_dist_indices, axis=1)

    # sort similarities decreasing order
    sorted_sims_indices = np.argsort(similarities, axis=1)[:, ::-1]

    gt_true = np.tile(
        np.triu(np.ones((n_samples, n_samples), dtype=bool)), (n_query, 1, 1)
    )

    gt_true = np.take_along_axis(gt_true, sorted_sims_indices[:, :, None], axis=1)
    true_positives = np.cumsum(gt_true, axis=1)
    all_positives = np.arange(n_samples)[None, :, None] + 1

    # compute precision and recall
    # all_positives > 0 always
    precision = true_positives / all_positives
    # by construction there is always at least one positive sample, so true_positives[-1] > 0
    recall = true_positives / true_positives[:, -1][:, None, :]

    avg_precisions = np.sum(
        np.diff(np.concatenate((np.zeros_like(recall[:, 0:1]), recall), axis=1), axis=1)
        * precision,
        axis=1,
    )

    # integrate over distance thresholds (weight by distance step size)
    dist_threshold_steps = np.diff(distances, axis=1)
    dist_threshold_steps = np.concatenate(
        (distances[:, 0:1], dist_threshold_steps), axis=1
    )

    mean_avg_precisions = (
        np.sum(avg_precisions * dist_threshold_steps, axis=1) / distances[:, -1]
    )

    return np.mean(mean_avg_precisions)


def mean_avg_precision(distances: torch.Tensor, similarities: torch.Tensor) -> float:
    """Compute the mean average precision, ie the average precision integrated over all distance thresholds, averaged over all query points.

    Parameters
    ----------
    distances (n_query x n_samples tensor):
        The 3D world distances between the (3D world points associated with the) query patch and the sample patches.

    similarities (n_query x n_samples tensor):
        The similarity scores between the query patch and the sample patches. Needs to be <

    Returns
    -------
    mean_avg_precision (float):
        The mean average precision, integrated over all distance thresholds, averaged over all query points.
    """
    n_query, n_samples = distances.shape
    device = distances.device

    # sort by similarity first to implicitly use as tie-breaker -- for _equal_ distances I give it the benefit of the doubt
    # not the _most_ efficient and probably almost never used, but I can't explicitly specify a tie-breaker in PyTorch to make it more efficient
    sort_idxs = torch.argsort(similarities, dim=1, descending=True)
    distances = torch.gather(distances, dim=1, index=sort_idxs)
    similarities = torch.gather(similarities, dim=1, index=sort_idxs)

    # Sort distances increasing order
    sorted_dist_indices = torch.argsort(distances, dim=1, stable=True)
    distances = torch.gather(distances, dim=1, index=sorted_dist_indices)
    similarities = torch.gather(similarities, dim=1, index=sorted_dist_indices)

    # Sort similarities decreasing order
    sorted_sims_indices = torch.argsort(similarities, dim=1, descending=True)
    sorted_sims_indices = sorted_sims_indices.unsqueeze(2).expand(-1, -1, n_samples)

    gt_true = torch.triu(
        torch.ones(n_query, n_samples, n_samples, dtype=torch.bool, device=device)
    )
    gt_true = torch.gather(gt_true, dim=1, index=sorted_sims_indices)

    true_positives = torch.cumsum(gt_true, dim=1).float()
    all_positives = torch.arange(n_samples, device=device, dtype=torch.float) + 1

    # Compute precision and recall
    # all_positives > 0 always
    precision = true_positives / all_positives[None, :, None]

    # by construction there is always at least one positive sample, so true_positives[-1] > 0
    recall = true_positives / true_positives[:, -1:]

    recall = torch.cat((torch.zeros_like(recall[:, 0:1]), recall), dim=1)
    avg_precisions = torch.sum((recall[:, 1:] - recall[:, :-1]) * precision, dim=1)

    # Integrate over distance thresholds (weight by distance step size)
    dist_threshold_steps = distances[:, 1:] - distances[:, :-1]
    dist_threshold_steps = torch.cat((distances[:, :1], dist_threshold_steps), dim=1)

    # this throws an error if all distancdes are 0, but such a case is likely to be a mistake in the input anyways
    mean_avg_precisions = (
        torch.sum(avg_precisions * dist_threshold_steps, dim=1) / distances[:, -1]
    )

    return torch.mean(mean_avg_precisions)


def pixel_correspondence_matching_recall(
    reproj_errors: torch.Tensor,
    pixel_thresholds: List[float] = [1, 5, 10, 20],
    angle_bins: List[float] = [0, 15, 30, 60, 180],
    gt_rotations: torch.Tensor = None,
) -> Dict[str, List[float]]:
    """Compute the recall of pixel correspondence matching for different pixel thresholds and angle bins, like in https://arxiv.org/pdf/2404.08636.pdf.

    Parameters
    ----------
    reproj_errors (n_correspondences torch.Tensor):
        The reprojection errors for a bunch of pixel correspondences.

    pixel_thresholds (List[float], optional):
        The pixel error thresholds to compute the recall for, by default [1, 5, 10, 20]

    angle_bins (List[float], optional):
        The boundaries of the bins for the relative rotations of the image pairs that the recall get computed for separately, by default [0, 15, 30, 60, 180]. Only used if gt_rotations is not None.

    gt_rotations (n_correspondences torch.Tensor or n_correspondences x 3 x 3 torch.Tensor):
        A tensor containing the ground truth relative rotations for each correspondence. Either the relative camera rotaion in degrees, or the relative rotation matrices. If None, the recall is only computed for all correspondences together.

    Returns
    -------
    Dict[str, List[float]]:
        A dictionary containing a list of the recall values for each pixel threshold, with the corresponding angle bin as key. The key is either 'all' or 'angle_bin_{lower}_to_{upper}'.
    """

    n_correspondences = reproj_errors.shape[0]

    if gt_rotations is not None:
        if gt_rotations.dim() > 1:
            rel_rotations = so3_rotation_angle(gt_rotations)
        else:
            rel_rotations = gt_rotations

    recalls = {}
    for pixel_threshold in pixel_thresholds:
        correct_matches = reproj_errors < pixel_threshold

        recalls[f"all_thr{pixel_threshold}px"] = (
            correct_matches.sum().item() / n_correspondences
        )

        if gt_rotations is not None:
            for i, angle_bin in enumerate(angle_bins[:-1]):
                lower, upper = angle_bin, angle_bins[i + 1]
                mask = (rel_rotations >= lower) & (rel_rotations < upper)
                recalls[f"angle_bin_{lower}_to_{upper}_thr{pixel_threshold}px"] = (
                    correct_matches[mask].sum().item() / max(mask.sum().item(), 0.0001)
                )

    return recalls
