"""Regression tests for the validation/training reference-count cross-check.

LtxTrainerConfig.validate_strategy_compatibility must reject configs where the
number of reference videos per validation prompt does not match the number of
training_strategy.reference_latents_dirs entries.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ltx_trainer.config import LtxTrainerConfig


def _config_dict(
    model_path: Path,
    ref_dirs: list[str],
    reference_videos: list[list[str]] | None,
    strategy: str = "video_to_video",
) -> dict:
    if strategy == "video_to_video":
        training_strategy = {"name": "video_to_video", "reference_latents_dirs": ref_dirs}
    else:
        training_strategy = {"name": "text_to_video"}
    validation = {"prompts": ["a prompt"], "interval": 100}
    if reference_videos is not None:
        validation["reference_videos"] = reference_videos
    return {
        "model": {"model_path": str(model_path), "training_mode": "lora"},
        "lora": {"rank": 8, "alpha": 8},
        "training_strategy": training_strategy,
        "data": {"preprocessed_data_root": str(model_path.parent)},
        "validation": validation,
    }


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"stub")
    return path


@pytest.fixture
def ref_files(tmp_path: Path) -> list[Path]:
    paths = [tmp_path / "ref0.mp4", tmp_path / "ref1.mp4"]
    for p in paths:
        p.write_bytes(b"stub")
    return paths


def test_matching_reference_count_passes(model_file: Path, ref_files: list[Path]) -> None:
    cfg = _config_dict(model_file, ref_dirs=["a", "b"], reference_videos=[[str(ref_files[0]), str(ref_files[1])]])
    LtxTrainerConfig(**cfg)  # must not raise


def test_mismatched_reference_count_raises(model_file: Path, ref_files: list[Path]) -> None:
    cfg = _config_dict(model_file, ref_dirs=["a", "b"], reference_videos=[[str(ref_files[0])]])
    with pytest.raises(ValidationError, match="reference\\(s\\) per prompt"):
        LtxTrainerConfig(**cfg)


def test_single_reference_backward_compat(model_file: Path, ref_files: list[Path]) -> None:
    cfg = _config_dict(model_file, ref_dirs=["reference_latents"], reference_videos=[[str(ref_files[0])]])
    LtxTrainerConfig(**cfg)  # legacy single-reference layout must still validate


def test_text_to_video_unaffected(model_file: Path) -> None:
    cfg = _config_dict(model_file, ref_dirs=[], reference_videos=None, strategy="text_to_video")
    LtxTrainerConfig(**cfg)  # cross-check only applies to video_to_video
