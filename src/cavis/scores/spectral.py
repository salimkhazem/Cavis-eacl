from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpectralScores:
    hfer: float
    fiedler: float
    smoothness: float
    spectral_entropy: float
    energy: float
    n_graph_tokens: int
    subsampled: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_float64_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("Spectral inputs must be finite")
    return result


def deterministic_token_indices(length: int, maximum: int) -> np.ndarray:
    """Select endpoints and evenly spaced interior tokens deterministically."""
    if length <= 0 or maximum <= 0:
        raise ValueError("length and maximum must be positive")
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, maximum, dtype=np.int64)


def attention_graph(
    attention: Any,
    *,
    mask: Any | None = None,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Turn directed multi-head attention into an undirected weighted graph.

    Accepted shapes are [heads,tokens,tokens] and [tokens,tokens]. Self loops
    are removed because they do not contribute to the combinatorial
    Laplacian. This definition is explicit and intentionally independent of
    unpublished implementation details of prior systems.
    """
    values = _as_float64_array(attention)
    if values.ndim == 3:
        head_mass = values.sum(axis=(1, 2))
        total_mass = float(head_mass.sum())
        if total_mass <= epsilon:
            weights = np.full(values.shape[0], 1.0 / values.shape[0])
        else:
            weights = head_mass / total_mass
        symmetrized = 0.5 * (values + values.transpose(0, 2, 1))
        values = np.einsum("h,hij->ij", weights, symmetrized)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("attention must be square [tokens,tokens] or [heads,tokens,tokens]")
    if mask is not None:
        keep = np.asarray(mask, dtype=bool)
        if keep.ndim != 1 or keep.shape[0] != values.shape[0]:
            raise ValueError("mask must have one entry per token")
        values = values[np.ix_(keep, keep)]
    values = np.maximum(values, 0.0)
    graph = 0.5 * (values + values.T)
    np.fill_diagonal(graph, 0.0)
    graph[graph < epsilon] = 0.0
    return graph


def normalized_laplacian(graph: Any, epsilon: float = 1e-12) -> np.ndarray:
    weights = _as_float64_array(graph)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("graph must be square")
    degree = weights.sum(axis=1)
    inverse_sqrt = np.zeros_like(degree)
    nonzero = degree > epsilon
    inverse_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    laplacian = np.eye(weights.shape[0]) - (
        inverse_sqrt[:, None] * weights * inverse_sqrt[None, :]
    )
    laplacian[~nonzero, ~nonzero] = 0.0
    return 0.5 * (laplacian + laplacian.T)


def combinatorial_laplacian(graph: Any) -> np.ndarray:
    """Return the unnormalized Laplacian used by Geometry of Reason."""

    weights = _as_float64_array(graph)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("graph must be square")
    laplacian = np.diag(weights.sum(axis=1)) - weights
    return 0.5 * (laplacian + laplacian.T)


def compute_spectral_scores(
    attention: Any,
    hidden_states: Any,
    *,
    mask: Any | None = None,
    high_frequency_fraction: float = 0.5,
    max_graph_tokens: int = 512,
    epsilon: float = 1e-12,
) -> SpectralScores:
    """Compute graph-frequency diagnostics from one layer.

    This follows the published Geometry of Reason definitions: attention-mass
    head aggregation, the combinatorial Laplacian, hidden-state graph Fourier
    energy, the median-frequency HFER cutoff by default, Dirichlet energy, and
    ``1 - E / (lambda_max ||X||_F^2)`` smoothness. Exact eigendecomposition is
    capped with a deterministic token subsample, which is recorded.
    """
    if not 0 < high_frequency_fraction <= 1:
        raise ValueError("high_frequency_fraction must be in (0,1]")
    states = _as_float64_array(hidden_states)
    if states.ndim != 2:
        raise ValueError("hidden_states must have shape [tokens,features]")

    attention_values = _as_float64_array(attention)
    token_count = attention_values.shape[-1]
    if states.shape[0] != token_count:
        raise ValueError("attention and hidden states must have the same token count")

    if mask is None:
        keep = np.ones(token_count, dtype=bool)
    else:
        keep = np.asarray(mask, dtype=bool)
        if keep.shape != (token_count,):
            raise ValueError("mask must have one entry per token")
    valid_indices = np.flatnonzero(keep)
    if valid_indices.size < 2:
        raise ValueError("At least two unmasked tokens are required")

    selected_relative = deterministic_token_indices(valid_indices.size, max_graph_tokens)
    selected = valid_indices[selected_relative]
    subsampled = selected.size < valid_indices.size
    states = states[selected]
    if attention_values.ndim == 3:
        attention_values = attention_values[:, selected][:, :, selected]
    elif attention_values.ndim == 2:
        attention_values = attention_values[np.ix_(selected, selected)]
    else:
        raise ValueError("attention must have two or three dimensions")

    graph = attention_graph(attention_values, epsilon=epsilon)
    laplacian = combinatorial_laplacian(graph)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    fiedler = float(eigenvalues[1]) if eigenvalues.size > 1 else 0.0

    state_energy = float(np.square(states).sum())
    dirichlet = float(np.einsum("tf,ts,sf->", states, laplacian, states))
    maximum_energy = float(eigenvalues[-1]) * state_energy
    smoothness = 1.0 - dirichlet / max(maximum_energy, epsilon)

    coefficients = eigenvectors.T @ states
    frequency_energy = np.square(coefficients).sum(axis=1)
    n_high = max(1, math.ceil(high_frequency_fraction * frequency_energy.size))
    total_frequency_energy = float(frequency_energy.sum())
    hfer = float(frequency_energy[-n_high:].sum() / max(total_frequency_energy, epsilon))
    if total_frequency_energy > epsilon:
        distribution = frequency_energy / total_frequency_energy
        entropy = float(
            -(distribution * np.log(distribution + epsilon)).sum()
        )
    else:
        entropy = 0.0

    return SpectralScores(
        hfer=hfer,
        fiedler=fiedler,
        smoothness=float(smoothness),
        spectral_entropy=entropy,
        energy=dirichlet,
        n_graph_tokens=int(selected.size),
        subsampled=subsampled,
    )
