"""CS-Fluctuation tracking for LoRA overfitting-onset detection.

Implements the diagnostic from "Towards Personalized AI: Early-stopping Low-Rank
Adaptation of Foundation Models" (CS-Fluctuation). For every LoRA layer it
measures the cosine similarity (CS) between the adapter's weight delta
(scaling * B @ A) and its frozen base weight W0, and averages across layers
(paper Eq. 2). It then tracks the *fluctuation* of that CS signal: the variance
of the smoothed CS slope, normalized by the learning rate (paper Eq. 3-4). When
the fluctuation becomes small, the CS has steadied -- the onset of overfitting.

Note on causality: the paper defines the moving-window average over a forward
window (j .. j+M), an offline computation. For online logging during training we
use a *trailing* window (j-M .. j) so the metric is available at step j; this
preserves the shape of the curve, shifted by the window. Because of the two
moving averages plus the variance window, the fluctuation only becomes available
after ~3*window measurements (it is omitted from the returned dict until then).

This is a logging-only diagnostic: it reads weights under no_grad and never
affects the loss, gradients, or the optimizer.
"""

from collections import deque

import torch
from peft.tuners.tuners_utils import BaseTunerLayer


class CSFluctuationTracker:
    """Track LoRA-vs-base cosine similarity and its CS-Fluctuation (paper Eq. 2-4).

    Each call to update() walks the LoRA layers of a PEFT model, computes the
    per-layer cosine between the adapter delta and the base weight, and returns
    the mean (plus min/max) across layers. It also maintains a rolling history of
    the mean CS and, once enough history has accumulated, reports the smoothed CS
    slope ("cs/slope") and the CS-Fluctuation ("cs/fluctuation").
    """

    def __init__(self, window: int = 20) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self._window = window
        # Eq. 4 chains two moving averages (window M) and a variance window (M),
        # so it needs ~3*M raw CS measurements before it can be evaluated.
        self._history: deque[float] = deque(maxlen=3 * window)

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

    @staticmethod
    def _trailing_ma(xs: list[float], m: int) -> list[float]:
        """Trailing moving average: out[t] = mean(xs[t-m+1 : t+1]) for t >= m-1."""
        if len(xs) < m:
            return []
        return [sum(xs[t - m + 1 : t + 1]) / m for t in range(m - 1, len(xs))]

    @classmethod
    def _fluctuation(cls, cs: list[float], m: int, lr: float) -> tuple[float | None, float | None]:
        """CS-Fluctuation (Eq. 3-4) and the current smoothed CS slope, causal form.

        Returns (fluctuation, slope). Either is None until enough history exists.
        ma1 = MA(CS); slope = grad(ma1); X = MA(slope); fluctuation = var(X[-m:]) / lr.
        """
        ma1 = cls._trailing_ma(cs, m)  # MA(CS)  -- Eq. 3
        if len(ma1) < 2:
            return None, None
        slope = [ma1[i] - ma1[i - 1] for i in range(1, len(ma1))]  # grad(MA(CS))
        x = cls._trailing_ma(slope, m)  # MA(grad(MA(CS))) -- smoothed slope
        cur_slope = slope[-1]
        if len(x) < m:
            return None, cur_slope
        win = x[-m:]
        mean_x = sum(win) / m
        var = sum((v - mean_x) ** 2 for v in win) / m  # Eq. 4 variance
        denom = abs(lr) if lr else 1.0  # normalize by lr (paper Eq. 4)
        return var / denom, cur_slope

    @torch.no_grad()
    def update(self, model: torch.nn.Module, lr: float = 1.0) -> dict[str, float]:
        """Measure mean LoRA-vs-base cosine across layers and its CS-Fluctuation.

        ``lr`` is the effective learning rate used to normalize the fluctuation
        (paper Eq. 4). Returns an empty dict when the model has no LoRA layers.
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
        fluctuation, slope = self._fluctuation(list(self._history), self._window, lr)
        if slope is not None:
            metrics["cs/slope"] = slope
        if fluctuation is not None:
            metrics["cs/fluctuation"] = fluctuation
        return metrics
