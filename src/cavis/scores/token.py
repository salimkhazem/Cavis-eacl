from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenScores:
    n_tokens: int
    mean_log_likelihood: float
    sequence_log_likelihood: float
    perplexity: float
    mean_entropy: float
    max_entropy: float
    variance_entropy: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_tokens": self.n_tokens,
            "mean_log_likelihood": self.mean_log_likelihood,
            "sequence_log_likelihood": self.sequence_log_likelihood,
            "perplexity": self.perplexity,
            "mean_entropy": self.mean_entropy,
            "max_entropy": self.max_entropy,
            "variance_entropy": self.variance_entropy,
        }


def token_scores(logits: Any, input_ids: Any, target_mask: Any | None = None) -> list[TokenScores]:
    """Compute teacher-forced token scores for a batch.

    `target_mask[:, t]` selects whether token `input_ids[:, t]` belongs to the
    reasoning span. Position zero cannot be scored and is always ignored.
    Imports torch lazily so CPU-only analysis does not require the HF extra.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in HF environments
        raise RuntimeError("token_scores requires the 'hf' optional dependencies") from exc

    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("Expected logits [batch,time,vocab] and input_ids [batch,time]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logit and input shapes do not agree")
    if input_ids.shape[1] < 2:
        raise ValueError("At least two tokens are required for next-token scoring")

    prediction_logits = logits[:, :-1].float()
    labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(prediction_logits, dim=-1)
    observed = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    probabilities = log_probs.exp()
    entropies = -(probabilities * log_probs).sum(dim=-1)

    if target_mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)
    else:
        if target_mask.shape != input_ids.shape:
            raise ValueError("target_mask must have the same shape as input_ids")
        mask = target_mask[:, 1:].to(dtype=torch.bool)

    results: list[TokenScores] = []
    for row in range(input_ids.shape[0]):
        selected_ll = observed[row][mask[row]]
        selected_entropy = entropies[row][mask[row]]
        if selected_ll.numel() == 0:
            raise ValueError(f"No target token selected for batch row {row}")
        mean_ll = selected_ll.mean()
        results.append(
            TokenScores(
                n_tokens=int(selected_ll.numel()),
                mean_log_likelihood=float(mean_ll.item()),
                sequence_log_likelihood=float(selected_ll.sum().item()),
                perplexity=float(torch.exp(-mean_ll).item()),
                mean_entropy=float(selected_entropy.mean().item()),
                max_entropy=float(selected_entropy.max().item()),
                variance_entropy=float(
                    selected_entropy.var(unbiased=False).item()
                ),
            )
        )
    return results

