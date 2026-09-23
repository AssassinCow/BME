from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from bme_eating.models.event_verifier import ProposalBatchSampler


DEFAULT_RATIOS = {
    "positive": 0.32,
    "near_miss": 0.24,
    "hard_false_positive": 0.24,
    "random_background": 0.20,
}


def _category_counts(categories: np.ndarray, batch: list[int]) -> Counter[str]:
    return Counter(categories[np.asarray(batch, dtype=int)])


def test_proposal_batch_sampler_enforces_registered_composition() -> None:
    categories = np.repeat(list(DEFAULT_RATIOS), 5)
    sampler = ProposalBatchSampler(categories, 100, DEFAULT_RATIOS, 2, seed=2026)

    batches = list(sampler)

    assert len(batches) == 2
    assert _category_counts(categories, batches[0]) == Counter(
        {
            "positive": 32,
            "near_miss": 24,
            "hard_false_positive": 24,
            "random_background": 20,
        }
    )


def test_proposal_batch_sampler_redistributes_empty_categories_in_order() -> None:
    categories = np.asarray(["positive", "random_background"])
    sampler = ProposalBatchSampler(categories, 10, DEFAULT_RATIOS, 1, seed=2026)

    batch = next(iter(sampler))

    assert _category_counts(categories, batch) == Counter(
        {"positive": 5, "random_background": 5}
    )


def test_proposal_batch_sampler_is_deterministic_per_epoch() -> None:
    categories = np.repeat(list(DEFAULT_RATIOS), 8)
    sampler = ProposalBatchSampler(categories, 16, DEFAULT_RATIOS, 3, seed=2026)

    first = list(sampler)
    second = list(sampler)
    sampler.set_epoch(1)
    third = list(sampler)

    assert first == second
    assert third != first


def test_proposal_batch_sampler_rejects_unknown_category() -> None:
    with pytest.raises(ValueError, match="Unknown proposal sampling categories"):
        ProposalBatchSampler(
            np.asarray(["positive", "unregistered"]),
            8,
            DEFAULT_RATIOS,
            1,
            seed=2026,
        )
