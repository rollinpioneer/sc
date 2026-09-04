"""Frozen verifier losses for Stage 3 v0.2-R.

There are intentionally no sampling or task weights here: the query-group
sampler defines the fixed four-stratum balance, and each sampled row receives
the same loss.  The ranking term is evaluated only for within-query target
gaps of at least ``rank_target_gap``.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def within_query_rank_loss(
    prediction: Tensor,
    target: Tensor,
    query_group_ids: Sequence[str],
    margin: float = 0.1,
    rank_target_gap: float = 0.2,
) -> Tensor:
    """Compute the fixed pairwise hinge ranking loss within query groups."""
    terms: list[Tensor] = []
    grouped: dict[str, list[int]] = {}
    for index, group_id in enumerate(query_group_ids):
        grouped.setdefault(str(group_id), []).append(index)
    for indices in grouped.values():
        for left, i in enumerate(indices):
            for j in indices[left + 1 :]:
                difference = target[i] - target[j]
                if abs(float(difference.detach())) < float(rank_target_gap):
                    continue
                direction = torch.sign(difference)
                terms.append(F.relu(prediction.new_tensor(float(margin)) - direction * (prediction[i] - prediction[j])))
    return torch.stack(terms).mean() if terms else prediction.new_zeros(())


def verifier_loss(
    prediction: dict[str, Tensor],
    batch: dict[str, Tensor],
    query_group_ids: Sequence[str],
    *,
    margin: float = 0.1,
    rank_target_gap: float = 0.2,
) -> dict[str, Tensor]:
    """Return regression, classification, rank, and total fixed losses."""
    regression = F.smooth_l1_loss(prediction["pred_score"], batch["target_score"])
    classification = F.binary_cross_entropy_with_logits(
        prediction["pred_positive_logit"], batch["target_positive"]
    )
    ranking = within_query_rank_loss(
        prediction["pred_score"],
        batch["target_score"],
        query_group_ids,
        margin=margin,
        rank_target_gap=rank_target_gap,
    )
    total = regression + 0.5 * classification + 0.25 * ranking
    return {
        "loss": total,
        "regression_loss": regression,
        "classification_loss": classification,
        "rank_loss": ranking,
    }
