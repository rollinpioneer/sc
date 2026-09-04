"""Losses for gated transition responsibility learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    return values[mask].mean() if bool(mask.any()) else values.new_zeros(())


def _available_weighted_mean(terms: list[tuple[float, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    numerator = None
    denominator = 0.0
    for weight, values, mask in terms:
        if bool(mask.bool().any()):
            term = float(weight) * values[mask.bool()].mean()
            numerator = term if numerator is None else numerator + term
            denominator += float(weight)
    if numerator is None:
        return terms[0][1].new_zeros(())
    return numerator / denominator


def responsibility_losses(output, batch: dict, *, lambda_localization: float = 1., lambda_effect: float = 1., lambda_rank: float = .25, rank_margin: float = .2, effect_loss_mode: str = "balanced_subtypes", effect_positive_mass: float = .5, effect_negative_mass: float = .5, negative_subtype_weights: dict | None = None, candidate_far_tolerance: float | None = None) -> dict[str, torch.Tensor]:
    logits, effect_logit = output.responsibility_logits, output.effect_logit
    target, effect = batch["target_distribution"].float(), batch["target_effect"].float()
    weight, valid = batch["sample_weight"].float(), batch["valid_mask"].bool()
    positive = effect > .5
    loc_per = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    loc = weighted_mean(loc_per, weight * positive.float())
    effect_per = F.binary_cross_entropy_with_logits(effect_logit, effect, reduction="none")
    if effect_loss_mode != "balanced_subtypes":
        raise ValueError(f"unsupported effect_loss_mode={effect_loss_mode}")
    groups = batch["effect_group_id"].long()
    positive_mask, no_effect_mask = groups == 0, groups == 1
    hard_negative_mask, clean_mask = groups == 2, groups == 3
    if not torch.equal(effect, positive_mask.float()):
        raise ValueError("target_effect and effect_group_id are inconsistent")
    subtype = negative_subtype_weights or {"no_effect_control": .45, "effective_hard_negative": .45, "clean_control": .10}
    positive_effect_loss = _masked_mean(effect_per, positive_mask)
    negative_effect_loss = _available_weighted_mean([
        (subtype["no_effect_control"], effect_per, no_effect_mask),
        (subtype["effective_hard_negative"], effect_per, hard_negative_mask),
        (subtype["clean_control"], effect_per, clean_mask),
    ])
    effect_loss = float(effect_positive_mass) * positive_effect_loss + float(effect_negative_mass) * negative_effect_loss
    positive_logit = (target * logits).sum(dim=-1)
    negative_mask = valid & (target == 0)
    hard_negative = logits.masked_fill(~negative_mask, torch.finfo(logits.dtype).min).max(dim=-1).values
    rank_per = torch.relu(rank_margin - positive_logit + hard_negative)
    rank = weighted_mean(rank_per, weight * positive.float())
    total = lambda_localization * loc + lambda_effect * effect_loss + lambda_rank * rank
    return {"loss": total, "loss_localization": loc.detach(), "loss_effect": effect_loss.detach(), "loss_effect_positive": positive_effect_loss.detach(), "loss_effect_negative": negative_effect_loss.detach(), "loss_effect_no_effect": _masked_mean(effect_per, no_effect_mask).detach(), "loss_effect_hard_negative": _masked_mean(effect_per, hard_negative_mask).detach(), "loss_effect_clean": _masked_mean(effect_per, clean_mask).detach(), "loss_rank": rank.detach()}
