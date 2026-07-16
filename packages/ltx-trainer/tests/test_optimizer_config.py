"""Tests for the prodigy_plus_schedulefree config validation.

OptimizationConfig.validate_schedulefree_scheduler — Schedule-Free requires a constant
scheduler, so any decaying scheduler must be rejected at config-validation time, and the
guard must not regress the other optimizers.
"""

import pytest
from pydantic import ValidationError

from ltx_trainer.config import OptimizationConfig


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
