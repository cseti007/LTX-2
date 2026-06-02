"""CS-Fluctuation tracking for LoRA overfitting-onset detection.

Implements the diagnostic from "Towards Personalized AI: Early-stopping Low-Rank
Adaptation of Foundation Models" (CS-Fluctuation). For every LoRA layer it
measures the cosine similarity between the adapter's weight delta
(scaling * B @ A) and its frozen base weight W0, averages across layers, and
tracks how much that average fluctuates over a rolling window of measurements.
When the fluctuation flattens out, it signals the onset of overfitting.

This is a logging-only diagnostic: it reads weights under no_grad and never
affects the loss, gradients, or the optimizer.
"""

from collections import deque

import torch
from peft.tuners.tuners_utils import BaseTunerLayer


class CSFluctuationTracker:
    """Track LoRA-vs-base cosine similarity and its rolling fluctuation.

    Each call to update() walks the LoRA layers of a PEFT model, computes the
    per-layer cosine between the adapter delta and the base weight, and returns
    the mean (plus min/max) across layers along with the standard deviation of
    the mean over the last `window` measurements ("CS-Fluctuation").
    """

    def __init__(self, window: int = 20) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self._history: deque[float] = deque(maxlen=window)

    @staticmethod
    def _layer_cosine(module: BaseTunerLayer, adapter: str) -> float | None:
        """Cosine similarity between the LoRA delta (scaling * B @ A) and W0.

        Computed without materializing the (out, in) delta, using:
            <delta, W0>  = scaling * sum(B * (W0 @ A^T))           # (out, r) intermediate
            ||delta||^2  = scaling^2 * sum((A @ A^T) * (B^T @ B))  # (r, r) intermediates
        """
        if adapter not in module.lora_A:
            return None
        a = module.lora_A[adapter].weight.float()  # (r, in)
        b = module.lora_B[adapter].weight.float()  # (out, r)
        w0 = module.base_layer.weight.float()  # (out, in)
        scaling = float(module.scaling[adapter])

        inner = scaling * (b * (w0 @ a.t())).sum()
        delta_norm = abs(scaling) * ((a @ a.t()) * (b.t() @ b)).sum().clamp_min(0).sqrt()
        denom = delta_norm * w0.norm()
        if denom == 0:
            return None
        return float(inner / denom)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> dict[str, float]:
        """Measure mean LoRA-vs-base cosine across layers and its fluctuation.

        Returns an empty dict when the model has no LoRA layers.
        """
        cosines: list[float] = []
        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                for adapter in module.active_adapters:
                    cos = self._layer_cosine(module, adapter)
                    if cos is not None:
                        cosines.append(cos)
        if not cosines:
            return {}

        mean_cos = sum(cosines) / len(cosines)
        self._history.append(mean_cos)
        metrics = {
            "cs/mean": mean_cos,
            "cs/min": min(cosines),
            "cs/max": max(cosines),
        }
        if len(self._history) >= 2:
            metrics["cs/fluctuation"] = float(torch.tensor(list(self._history)).std(unbiased=False))
        return metrics
