"""Regression tests for CS-Fluctuation logging.

Covers two layers:
  - CSFluctuationTracker: the cosine identity vs a naive reference, boundary
    cases, and the empty/perturbed behaviour on a real PEFT model.
  - LtxvTrainer._compute_cs_fluctuation: the trainer hook gating (enabled,
    interval, FSDP skip) exercised against a stub `self`.
"""

from types import SimpleNamespace

import pytest
import torch
from accelerate import DistributedType
from peft import LoraConfig, get_peft_model
from torch import nn

from ltx_trainer.config import CSFluctuationConfig
from ltx_trainer.cs_fluctuation_tracker import CSFluctuationTracker
from ltx_trainer.trainer import LtxvTrainer


def _layer(a: torch.Tensor, b: torch.Tensor, w0: torch.Tensor, scaling: float) -> SimpleNamespace:
    """Build a minimal object matching the PEFT BaseTunerLayer attributes the tracker reads."""
    return SimpleNamespace(
        lora_A={"default": SimpleNamespace(weight=a)},
        lora_B={"default": SimpleNamespace(weight=b)},
        base_layer=SimpleNamespace(weight=w0),
        scaling={"default": scaling},
    )


def _naive_cosine(a: torch.Tensor, b: torch.Tensor, w0: torch.Tensor, scaling: float) -> float:
    """Reference cosine(scaling * B @ A, W0) computed by materializing the delta."""
    delta = scaling * (b @ a)
    return float(
        torch.dot(delta.reshape(-1).float(), w0.reshape(-1).float()) / (delta.norm() * w0.norm())
    )


def _tiny_lora_model(noise_std: float = 0.0) -> nn.Module:
    """A small PEFT model; perturb lora_B so the delta is non-zero when noise_std > 0."""

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = nn.Linear(32, 32, bias=False)
            self.v = nn.Linear(32, 32, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.v(self.q(x))

    model = get_peft_model(Net(), LoraConfig(r=4, lora_alpha=4, target_modules=["q", "v"]))
    if noise_std > 0:
        for module in model.modules():
            if hasattr(module, "lora_B"):
                for adapter in module.lora_B:
                    module.lora_B[adapter].weight.data.normal_(0, noise_std)
    return model


# --- CSFluctuationTracker: the cosine math -----------------------------------


def test_cosine_identity_matches_naive() -> None:
    """The memory-efficient identity must match the naive cosine across shapes/scalings."""
    torch.manual_seed(0)
    max_err = 0.0
    for _ in range(100):
        out, inn, r = torch.randint(1, 48, (3,)).tolist()
        a = torch.randn(r, inn)
        b = torch.randn(out, r)
        w0 = torch.randn(out, inn)
        scaling = float(torch.rand(1) * 4 - 2)  # include negative scaling
        fast = CSFluctuationTracker._layer_cosine(_layer(a, b, w0, scaling), "default")
        max_err = max(max_err, abs(fast - _naive_cosine(a, b, w0, scaling)))
    assert max_err < 1e-4, f"identity diverged from naive by {max_err}"


def test_cosine_boundary_cases() -> None:
    """delta == W0 -> cosine 1; delta == -W0 -> cosine -1."""
    a = torch.randn(4, 20)
    b = torch.randn(25, 4)
    assert CSFluctuationTracker._layer_cosine(_layer(a, b, b @ a, 1.0), "default") == pytest.approx(1.0, abs=1e-4)
    assert CSFluctuationTracker._layer_cosine(_layer(a, b, -(b @ a), 1.0), "default") == pytest.approx(-1.0, abs=1e-4)


def test_tracker_fresh_lora_returns_empty() -> None:
    """A freshly initialized LoRA (B == 0) has a zero delta, so layers are skipped."""
    tracker = CSFluctuationTracker(window=5)
    assert tracker.update(_tiny_lora_model(noise_std=0.0)) == {}


def test_tracker_returns_metrics_and_fluctuation() -> None:
    """After perturbation, metrics are present, in range, and fluctuation appears after >=2 updates."""
    model = _tiny_lora_model(noise_std=0.1)
    tracker = CSFluctuationTracker(window=5)

    first = tracker.update(model)
    assert {"cs/mean", "cs/min", "cs/max"}.issubset(first)
    assert -1.0001 <= first["cs/min"] <= first["cs/max"] <= 1.0001
    assert "cs/fluctuation" not in first  # needs at least two measurements

    second = tracker.update(model)
    assert "cs/fluctuation" in second
    assert second["cs/fluctuation"] >= 0.0


def test_tracker_rejects_small_window() -> None:
    with pytest.raises(ValueError, match="window must be >= 2"):
        CSFluctuationTracker(window=1)


# --- LtxvTrainer._compute_cs_fluctuation: the trainer hook gating -------------


def _stub_trainer(enabled: bool, interval: int, step: int, dist: DistributedType, model: nn.Module) -> SimpleNamespace:
    accelerator = SimpleNamespace(distributed_type=dist, unwrap_model=lambda m: m)
    return SimpleNamespace(
        _config=SimpleNamespace(cs_fluctuation=CSFluctuationConfig(enabled=enabled, interval=interval, window=5)),
        _global_step=step,
        _accelerator=accelerator,
        _transformer=model,
        _cs_tracker=CSFluctuationTracker(window=5),
        _cs_fsdp_warned=False,
    )


def test_hook_disabled_returns_empty() -> None:
    stub = _stub_trainer(enabled=False, interval=10, step=10, dist=DistributedType.NO, model=_tiny_lora_model(0.1))
    assert LtxvTrainer._compute_cs_fluctuation(stub) == {}


def test_hook_off_interval_returns_empty() -> None:
    stub = _stub_trainer(enabled=True, interval=10, step=5, dist=DistributedType.NO, model=_tiny_lora_model(0.1))
    assert LtxvTrainer._compute_cs_fluctuation(stub) == {}


def test_hook_on_interval_returns_metrics() -> None:
    stub = _stub_trainer(enabled=True, interval=10, step=10, dist=DistributedType.NO, model=_tiny_lora_model(0.1))
    metrics = LtxvTrainer._compute_cs_fluctuation(stub)
    assert {"cs/mean", "cs/min", "cs/max"}.issubset(metrics)


def test_hook_fsdp_skips_and_warns_once() -> None:
    stub = _stub_trainer(enabled=True, interval=10, step=10, dist=DistributedType.FSDP, model=_tiny_lora_model(0.1))
    assert LtxvTrainer._compute_cs_fluctuation(stub) == {}
    assert stub._cs_fsdp_warned is True
