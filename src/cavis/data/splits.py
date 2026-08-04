"""Seeded, leakage-resistant grouping and sampling."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TypeVar

T = TypeVar("T")


def stable_hash(value: object, seed: int = 0, namespace: str = "cavis") -> int:
    """Return a process-independent 256-bit hash.

    Python's builtin ``hash`` is intentionally randomized between processes,
    so it must never be used for experiment partitions.
    """

    payload = f"{namespace}\0{seed}\0{value}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big", signed=False)


def assign_grouped_splits(
    group_ids: Iterable[str],
    *,
    seed: int,
    ratios: Mapping[str, float] | None = None,
) -> dict[str, str]:
    """Assign every distinct group to one split using stable hash thresholds.

    This method is invariant to row ordering and to adding unrelated groups.
    Ratios need not sum to exactly one due to decimal representation, but must
    be within floating-point tolerance.
    """

    split_ratios = dict(ratios or {"train": 0.6, "calibration": 0.2, "test": 0.2})
    if not split_ratios:
        raise ValueError("At least one split is required")
    if any(value < 0 for value in split_ratios.values()):
        raise ValueError("Split ratios must be non-negative")
    total = sum(split_ratios.values())
    if abs(total - 1.0) > 1e-12:
        raise ValueError(f"Split ratios must sum to 1, got {total}")

    cumulative: list[tuple[str, float]] = []
    running = 0.0
    for name, fraction in split_ratios.items():
        running += fraction
        cumulative.append((name, running))
    cumulative[-1] = (cumulative[-1][0], 1.0)

    assignments: dict[str, str] = {}
    denominator = float(1 << 256)
    for group_id in sorted(set(group_ids)):
        unit = stable_hash(group_id, seed=seed, namespace="split") / denominator
        assignments[group_id] = next(
            name for name, upper_bound in cumulative if unit < upper_bound
        )
    return assignments


def deterministic_sample(
    items: Sequence[T],
    n: int,
    *,
    seed: int,
    key: Callable[[T], object],
    strata: Callable[[T], object] | None = None,
) -> list[T]:
    """Select ``n`` items deterministically, optionally balancing strata.

    For stratified sampling, seats are allocated round-robin across sorted
    strata after each stratum is independently hash-shuffled.  This avoids
    dependence on input ordering and keeps stratum counts within one whenever
    all strata contain enough examples.
    """

    if n < 0:
        raise ValueError("n must be non-negative")
    if n >= len(items):
        return sorted(
            items,
            key=lambda item: (
                stable_hash(key(item), seed=seed, namespace="sample"),
                str(key(item)),
            ),
        )
    if n == 0:
        return []

    rank = lambda item: (  # noqa: E731 - local deterministic ranking function
        stable_hash(key(item), seed=seed, namespace="sample"),
        str(key(item)),
    )
    if strata is None:
        return sorted(items, key=rank)[:n]

    buckets: dict[str, list[T]] = defaultdict(list)
    for item in items:
        buckets[str(strata(item))].append(item)
    for values in buckets.values():
        values.sort(key=rank)

    selected: list[T] = []
    names = sorted(buckets)
    cursor = {name: 0 for name in names}
    while len(selected) < n:
        progressed = False
        for name in names:
            index = cursor[name]
            if index < len(buckets[name]) and len(selected) < n:
                selected.append(buckets[name][index])
                cursor[name] += 1
                progressed = True
        if not progressed:  # defensive; n <= len(items) makes this unreachable
            break
    return selected


def proportional_stratified_sample(
    items: Sequence[T],
    n: int,
    *,
    seed: int,
    key: Callable[[T], object],
    strata: Callable[[T], object],
) -> list[T]:
    """Hash-sample while preserving stratum prevalence up to integer rounding."""

    if n < 0:
        raise ValueError("n must be non-negative")
    if n >= len(items):
        return deterministic_sample(items, len(items), seed=seed, key=key)
    if n == 0:
        return []
    buckets: dict[str, list[T]] = defaultdict(list)
    for item in items:
        buckets[str(strata(item))].append(item)
    rank = lambda item: (  # noqa: E731
        stable_hash(key(item), seed=seed, namespace="proportional-sample"),
        str(key(item)),
    )
    for values in buckets.values():
        values.sort(key=rank)

    total = len(items)
    exact = {name: n * len(values) / total for name, values in buckets.items()}
    seats = {name: int(value) for name, value in exact.items()}
    remaining = n - sum(seats.values())
    priority = sorted(
        buckets,
        key=lambda name: (
            -(exact[name] - seats[name]),
            stable_hash(name, seed=seed, namespace="stratum-tie"),
            name,
        ),
    )
    while remaining:
        progressed = False
        for name in priority:
            if seats[name] < len(buckets[name]) and remaining:
                seats[name] += 1
                remaining -= 1
                progressed = True
        if not progressed:  # defensive
            break
    selected = [
        item
        for name in sorted(buckets)
        for item in buckets[name][: seats[name]]
    ]
    return sorted(selected, key=rank)
