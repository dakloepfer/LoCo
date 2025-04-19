import torch
from einops import rearrange


class VecSmoothAP(torch.nn.Module):
    """Computes the Vectorised Smooth Average Precision loss."""

    def __init__(self, sigmoid_temperature):
        super().__init__()

        self.sigmoid_temperature = sigmoid_temperature

    def forward(
        self,
        anchor_similarities,
        non_anchor_pos_similarities,
        non_anchor_neg_similarities,
        batch,
    ):
        """Computes the Vectorised Smooth Average Precision loss given similarities for anchor pairs, similarities for the corresponding sampled non-anchor pairs, and a number of other correction variables contained in the batch dictionary.

        Parameters
        ----------
        anchor_similarities (n_anchor_pairs tensor):
            the similarities of the anchor pairs

        non_anchor_pos_similarities (n_anchor_pairs x n_pos_non_anchor_pairs tensor):
            the similarities of the corresponding non-anchor positive pairs

        non_anchor_neg_similarities (n_anchor_pairs x n_neg_non_anchor_pairs tensor):
            the similarities of the corresponding non-anchor negative pairs

        batch (dict):
            minimal keys:
                valid_nonanchor_pos_pair_mask (n_anchor_pairs x n_pos_non_anchor_pairs tensor):
                    a mask indicating which of the positive non-anchor pairs are valid (True for valid pairs, False for invalid pairs)

                valid_nonanchor_neg_pair_mask (n_anchor_pairs x n_neg_non_anchor_pairs tensor):
                    a mask indicating which of the negative non-anchor pairs are valid (True for valid pairs, False for invalid pairs)

                batch_correction_factor_positive (n_anchor_pairs tensor):
                    the batch correction factor for the sum over positive pairs in the loss function for each anchor pair

                batch_correction_factor_negative (n_anchor_pairs tensor):
                    the batch correction factor for the sum over negative pairs in the loss function for each anchor pair

                sampling_correction_factor_positive (n_anchor_pairs tensor):
                    the sampling correction term for the sum over positive pairs in the loss function for each anchor pair

                sampling_correction_factor_negative (n_anchor_pairs tensor):
                    the sampling correction term for the sum over negative pairs in the loss function for each anchor pair

                n_total_positive_pairs (scalar tensor):
                    the total number of positive pairs in the scene

        Returns
        -------
        loss (scalar tensor):
            -Vectorised Smooth AP as the loss

        log_dict (dict):
            a dictionary with information to log
        """
        # Collect all necessary variables from the batch dictionary
        valid_nonanchor_pos_pair_mask = batch["valid_nonanchor_pos_pair_mask"]
        valid_nonanchor_neg_pair_mask = batch["valid_nonanchor_neg_pair_mask"]
        batch_correction_factor_positive = batch["batch_correction_factor_positive"]
        batch_correction_factor_negative = batch["batch_correction_factor_negative"]
        sampling_correction_factor_positive = batch[
            "sampling_correction_factor_positive"
        ]
        sampling_correction_factor_negative = batch[
            "sampling_correction_factor_negative"
        ]
        n_total_positive_pairs = batch["n_total_positive_pairs"]

        # calculate similarity differences
        pos_sim_diffs = non_anchor_pos_similarities - anchor_similarities[:, None]
        pos_sim_diffs = torch.sigmoid(pos_sim_diffs / self.sigmoid_temperature)
        pos_sim_diffs = pos_sim_diffs * valid_nonanchor_pos_pair_mask
        neg_sim_diffs = non_anchor_neg_similarities - anchor_similarities[:, None]
        neg_sim_diffs = torch.sigmoid(neg_sim_diffs / self.sigmoid_temperature)
        neg_sim_diffs = neg_sim_diffs * valid_nonanchor_neg_pair_mask

        # calculate the numerator and denominator of the loss function
        numerator = (
            1 / (n_total_positive_pairs - 1)
            + batch_correction_factor_positive * pos_sim_diffs.sum(-1)
            + sampling_correction_factor_positive
        )
        denominator = (
            numerator
            + batch_correction_factor_negative * neg_sim_diffs.sum(-1)
            + sampling_correction_factor_negative
        )

        anchor_pair_smooth_aps = numerator / denominator

        loss = -anchor_pair_smooth_aps.mean()

        log_data = {
            "anchor_pair_smooth_ap": anchor_pair_smooth_aps.cpu(),
            "numerator": numerator.cpu(),
            "denominator": denominator.cpu(),
            "pos_sim_diffs": pos_sim_diffs[valid_nonanchor_pos_pair_mask].cpu(),
            "neg_sim_diffs": neg_sim_diffs[valid_nonanchor_neg_pair_mask].cpu(),
            "batch_correction_factor_positive": batch_correction_factor_positive.cpu(),
            "batch_correction_factor_negative": batch_correction_factor_negative.cpu(),
            "sampling_correction_factor_positive": sampling_correction_factor_positive.cpu(),
            "sampling_correction_factor_negative": sampling_correction_factor_negative.cpu(),
            "anchor_similarities": anchor_similarities.cpu(),
            "non_anchor_pos_similarities": non_anchor_pos_similarities[
                valid_nonanchor_pos_pair_mask
            ].cpu(),
            "non_anchor_neg_similarities": non_anchor_neg_similarities[
                valid_nonanchor_neg_pair_mask
            ].cpu(),
            "n_valid_nonanchor_pos_pairs": valid_nonanchor_pos_pair_mask.sum(-1).cpu(),
            "n_valid_nonanchor_neg_pairs": valid_nonanchor_neg_pair_mask.sum(-1).cpu(),
            "frac_pos_nonanchor_pairs": (
                valid_nonanchor_pos_pair_mask.sum(-1)
                / (
                    valid_nonanchor_pos_pair_mask.sum(-1)
                    + valid_nonanchor_neg_pair_mask.sum(-1)
                )
            ).cpu(),
            "n_anchor_pairs": float(anchor_similarities.shape[0]),
        }

        return loss, log_data
