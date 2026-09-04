"""One-batch integrity smoke for the Stage 2 responsibility dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stage2.data.responsibility_dataset import ResponsibilityDataset, make_dataloader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-index", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    dataset = ResponsibilityDataset(args.chunk_index, args.normalizer, max_horizon=int(config["data"]["max_horizon"]), split="train")
    batch = next(iter(make_dataloader(dataset, args.batch_size, train=True)))
    assert batch["image"].shape[-1] == 768 and batch["action_features"].shape[-1] == 23
    assert batch["image"].shape[0] > 0 and torch.all(batch["valid_mask"].sum(dim=1) >= 8)
    assert "effect_group_id" in batch
    assert set(batch["effect_group_id"].cpu().tolist()).issubset({0, 1, 2, 3})
    assert torch.equal(batch["target_effect"].long(), (batch["effect_group_id"] == 0).long())
    pos = batch["target_effect"] == 1
    assert torch.allclose(batch["target_distribution"][pos].sum(dim=1), torch.ones(pos.sum()), atol=1e-5)
    assert torch.equal(batch["target_distribution"][~pos].sum(dim=1), torch.zeros((~pos).sum()))
    # A small sampled batch can legitimately contain no positives because the
    # documented pair weighting is applied after class weighting. Validate one
    # deterministic positive directly so the region target is always covered.
    positive_index = int(dataset.samples.index[dataset.samples.target_effect == 1][0])
    positive_item = dataset[positive_index]
    assert torch.isclose(positive_item["target_distribution"].sum(), torch.tensor(1.0), atol=1e-5)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value.float()).all(), key
    print({"batch_size": int(batch["image"].shape[0]), "positive_rows": int(pos.sum()), "smoke": "passed"})


if __name__ == "__main__":
    main()
