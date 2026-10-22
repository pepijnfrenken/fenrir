"""Prompt partitions must represent the source's positional strata."""

from collections import Counter

import pytest

from eval_split import build_split


@pytest.mark.parametrize("partition,per_stratum", [("train", 8), ("tune", 4), ("test", 4)])
def test_each_partition_covers_every_stratum(partition, per_stratum):
    harmful = [f"harmful-{i}" for i in range(80)]
    harmless = [f"harmless-{i}" for i in range(80)]
    split = build_split(harmful, harmless, strata=5)

    counts = Counter(int(prompt.split("-")[1]) // 16 for prompt in getattr(split, partition))
    assert counts == {i: per_stratum for i in range(5)}


def test_uneven_strata_preserve_sizes_and_disjoint_deterministic_partitions():
    harmful = [f"harmful-{i}" for i in range(17)]
    harmless = [f"harmless-{i}" for i in range(17)]
    kwargs = dict(train_size=9, tune_size=4, test_size=4, strata=4, seed="regression")
    first = build_split(harmful, harmless, **kwargs)
    second = build_split(harmful, harmless, **kwargs)

    assert first == second
    assert (len(first.train), len(first.tune), len(first.test)) == (9, 4, 4)
    assert set(first.train).isdisjoint(first.tune)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.tune).isdisjoint(first.test)
    assert set(first.train + first.tune + first.test) == set(harmful)
    for partition in (first.tune, first.test):
        counts = Counter(int(prompt.split("-")[1]) * 4 // 17 for prompt in partition)
        assert counts == {i: 1 for i in range(4)}


def test_more_strata_than_prompts_handles_empty_strata():
    split = build_split(["a", "b"], ["x", "y"], train_size=1, tune_size=0, test_size=1, strata=5)
    assert len(split.train) == len(split.test) == 1
    assert set(split.train).isdisjoint(split.test)


def test_empty_split():
    split = build_split([], [], train_size=0, tune_size=0, test_size=0)
    assert split.train == split.tune == split.test == ()
