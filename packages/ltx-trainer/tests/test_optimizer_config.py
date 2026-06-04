"""Tests for the prodigy_plus_schedulefree optimizer integration.

Covers two things:
1. OptimizationConfig.validate_schedulefree_scheduler — Schedule-Free requires a constant
   scheduler, so any decaying scheduler must be rejected at config-validation time.
2. LtxvTrainer._optimizer_eval/_optimizer_train — the duck-typing mode switch must be a no-op
   for optimizers without train()/eval() (AdamW, Prodigy) and must call them when present
   (ProdigyPlusScheduleFree).
"""

import pytest
from pydantic import ValidationError

from ltx_trainer.config import OptimizationConfig
from ltx_trainer.trainer import LtxvTrainer


def test_schedulefree_requires_constant_scheduler() -> None:
    with pytest.raises(ValidationError, match="requires scheduler_type='constant'"):
        OptimizationConfig(
            optimizer_type="prodigy_plus_schedulefree",
            scheduler_type="linear",
        )


def test_schedulefree_with_constant_scheduler_ok() -> None:
    cfg = OptimizationConfig(
        optimizer_type="prodigy_plus_schedulefree",
        scheduler_type="constant",
        learning_rate=1.0,
    )
    assert cfg.optimizer_type == "prodigy_plus_schedulefree"


def test_other_optimizers_allow_decaying_scheduler() -> None:
    # The scheduler guard must only apply to prodigy_plus_schedulefree — no regression for the rest.
    for optimizer_type in ("adamw", "adamw8bit", "prodigy"):
        cfg = OptimizationConfig(optimizer_type=optimizer_type, scheduler_type="linear")
        assert cfg.scheduler_type == "linear"


def test_optimizer_mode_helpers_noop_without_methods() -> None:
    class _PlainOptimizer:
        """Stands in for AdamW/Prodigy — no train()/eval()."""

    trainer = object.__new__(LtxvTrainer)
    trainer._optimizer = _PlainOptimizer()
    # Must not raise even though the optimizer has no train()/eval().
    trainer._optimizer_eval()
    trainer._optimizer_train()


def test_optimizer_mode_helpers_call_when_present() -> None:
    calls: list[str] = []

    class _ScheduleFreeOptimizer:
        def eval(self) -> None:
            calls.append("eval")

        def train(self) -> None:
            calls.append("train")

    trainer = object.__new__(LtxvTrainer)
    trainer._optimizer = _ScheduleFreeOptimizer()
    trainer._optimizer_eval()
    trainer._optimizer_train()
    assert calls == ["eval", "train"]
