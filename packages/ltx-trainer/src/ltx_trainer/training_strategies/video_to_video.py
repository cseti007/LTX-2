"""Video-to-video training strategy for IC-LoRA.
This strategy implements training with reference video conditioning where:
- Reference latents (clean) are concatenated with target latents (noised)
- Video coordinates handle both reference and target sequences
- Loss is computed only on the target portion
"""

from typing import Any, Literal

import torch
from pydantic import Field, model_validator
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
    _broadcast_sigma,
    _sample_noise_like,
)


class VideoToVideoConfig(TrainingStrategyConfigBase):
    """Configuration for video-to-video (IC-LoRA) training strategy."""

    name: Literal["video_to_video"] = "video_to_video"

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    reference_latents_dirs: list[str] = Field(
        default_factory=lambda: ["reference_latents"],
        description=(
            "Directory names for latents of reference videos. All references are "
            "concatenated in this order before the target. Use a single-element list "
            "for standard IC-LoRA training, or multiple elements for multi-reference "
            "conditioning. All references must share the same spatial resolution "
            "(height × width); frame counts may differ across references."
        ),
        min_length=1,
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_reference_latents_dir(cls, data: Any) -> Any:
        # BC for pre-N-way configs: `reference_latents_dir: "foo"` (str) was
        # replaced by `reference_latents_dirs: ["foo"]` (list[str]). Accept the
        # legacy field by rewriting it into the new one before standard
        # validation, so existing YAMLs keep loading. The new field wins if both
        # are present.
        if isinstance(data, dict) and "reference_latents_dir" in data:
            legacy = data.pop("reference_latents_dir")
            if "reference_latents_dirs" not in data:
                data["reference_latents_dirs"] = [legacy]
        return data


class VideoToVideoStrategy(TrainingStrategy):
    """Video-to-video training strategy for IC-LoRA.
    This strategy implements training with reference video conditioning where:
    - Reference latents (clean) are concatenated with target latents (noised)
    - Video coordinates handle both reference and target sequences
    - Loss is computed only on the target portion
    Attributes:
        reference_downscale_factor: The inferred downscale factor of reference videos.
            This is computed from the first batch and cached for metadata export.
    """

    config: VideoToVideoConfig
    reference_downscale_factor: int | None

    def __init__(self, config: VideoToVideoConfig):
        """Initialize strategy with configuration.
        Args:
            config: Video-to-video configuration
        """
        super().__init__(config)
        self.reference_downscale_factor = None  # Will be inferred from first batch

    def get_data_sources(self) -> dict[str, str]:
        """IC-LoRA training requires latents, conditions, and reference latents.

        For multi-reference training, each entry in ``reference_latents_dirs`` is
        exposed under a distinct batch key ``ref_latents_{i}`` (0-indexed).
        """
        sources = {
            "latents": "latents",
            "conditions": "conditions",
        }
        for i, ref_dir in enumerate(self.config.reference_latents_dirs):
            sources[ref_dir] = f"ref_latents_{i}"
        return sources

    def prepare_training_inputs(  # noqa: PLR0915
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
        *,
        override_sigma: float | Tensor | None = None,
        noise_generator: torch.Generator | None = None,
    ) -> ModelInputs:
        """Prepare inputs for IC-LoRA training with reference videos."""
        # Get pre-encoded latents - dataset provides uniform non-patchified format [B, C, F, H, W]
        latents = batch["latents"]
        target_latents = latents["latents"]

        # Load N reference latents (ref_latents_0, ref_latents_1, ...) produced by get_data_sources()
        num_refs = len(self.config.reference_latents_dirs)
        ref_infos = [batch[f"ref_latents_{i}"] for i in range(num_refs)]
        ref_latents_list = [info["latents"] for info in ref_infos]

        # Get target dimensions
        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        # Per-reference (frames, height, width). Frame counts may differ across refs
        # (e.g. a short gradient-signal ref + a longer content-pool ref), but the
        # spatial resolution must be identical so a single downscale factor applies.
        ref_dims_list = [
            (info["num_frames"][0].item(), info["height"][0].item(), info["width"][0].item())
            for info in ref_infos
        ]
        ref_height = ref_dims_list[0][1]
        ref_width = ref_dims_list[0][2]
        for i, (_f_i, h_i, w_i) in enumerate(ref_dims_list[1:], start=1):
            if (h_i, w_i) != (ref_height, ref_width):
                raise ValueError(
                    f"All reference videos must share the same spatial resolution. "
                    f"Reference 0 has (height={ref_height}, width={ref_width}), "
                    f"but reference {i} has (height={h_i}, width={w_i})."
                )

        # Infer reference downscale factor from dimension ratios
        # This allows training with downscaled reference videos for efficiency
        reference_downscale_factor = self._infer_reference_downscale_factor(
            target_height=height,
            target_width=width,
            ref_height=ref_height,
            ref_width=ref_width,
        )

        # Cache the scale factor for metadata export (only on first batch)
        if self.reference_downscale_factor is None:
            self.reference_downscale_factor = reference_downscale_factor
        elif self.reference_downscale_factor != reference_downscale_factor:
            raise ValueError(
                f"Inconsistent reference downscale factor across batches. "
                f"First batch had factor={self.reference_downscale_factor}, "
                f"but current batch has factor={reference_downscale_factor}. "
                f"All training samples must use the same reference/target resolution ratio."
            )

        # Patchify latents: [B, C, F, H, W] -> [B, seq_len, C]
        target_latents = self._video_patchifier.patchify(target_latents)
        ref_latents_list = [self._video_patchifier.patchify(r) for r in ref_latents_list]

        # Handle FPS
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get text embeddings (already processed by embedding connectors in trainer)
        # Video-to-video uses only video embeddings
        conditions = batch["conditions"]
        prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = target_latents.shape[0]
        # Each reference contributes its own sequence; total_ref_seq_len is the sum
        # (used for loss slicing, mask sizing, and the target-region offset).
        ref_seq_lens = [r.shape[1] for r in ref_latents_list]
        total_ref_seq_len = sum(ref_seq_lens)
        target_seq_len = target_latents.shape[1]
        device = target_latents.device
        dtype = target_latents.dtype

        # Create conditioning mask
        # Reference tokens are always conditioning (timestep=0)
        ref_conditioning_mask = torch.ones(batch_size, total_ref_seq_len, dtype=torch.bool, device=device)

        # Target tokens: check for first frame conditioning
        target_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=target_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        # Combined conditioning mask
        conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)

        # Sample noise and sigmas for target (or use deterministic overrides for val loss)
        if override_sigma is None:
            sigmas = timestep_sampler.sample_for(target_latents)
        else:
            sigmas = _broadcast_sigma(override_sigma, batch_size, device, target_latents.dtype)
        noise = _sample_noise_like(target_latents, noise_generator)
        sigmas_expanded = sigmas.view(-1, 1, 1)

        # Apply noise to target
        noisy_target = (1 - sigmas_expanded) * target_latents + sigmas_expanded * noise

        # For first frame conditioning in target, use clean latents
        target_conditioning_mask_expanded = target_conditioning_mask.unsqueeze(-1)
        noisy_target = torch.where(target_conditioning_mask_expanded, target_latents, noisy_target)

        # Targets for loss computation
        targets = noise - target_latents

        # Concatenate all references (clean) in order, then target (noisy)
        combined_latents = torch.cat([*ref_latents_list, noisy_target], dim=1)

        # Create per-token timesteps
        timesteps = self._create_per_token_timesteps(conditioning_mask, sigmas.squeeze())

        # Generate positions per reference (each may have its own frame count),
        # then concatenate along the sequence dimension together with target positions.
        # Position tensor shape per call: [B, 3, seq_len, 2] where dim 1 is (time, height, width).
        ref_positions_list = []
        for f_i, h_i, w_i in ref_dims_list:
            pos = self._get_video_positions(
                num_frames=f_i,
                height=h_i,
                width=w_i,
                batch_size=batch_size,
                fps=fps,
                device=device,
                dtype=dtype,
            )
            # Scale reference positions to match target coordinate space (spatial only);
            # time axis (index 0) is left untouched.
            if reference_downscale_factor != 1:
                pos = pos.clone()
                pos[:, 1, ...] *= reference_downscale_factor  # height axis
                pos[:, 2, ...] *= reference_downscale_factor  # width axis
            ref_positions_list.append(pos)

        all_ref_positions = torch.cat(ref_positions_list, dim=2) if num_refs > 1 else ref_positions_list[0]

        target_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        # Concatenate positions along sequence dimension
        positions = torch.cat([all_ref_positions, target_positions], dim=2)

        # Create video Modality
        video_modality = Modality(
            enabled=True,
            latent=combined_latents,
            sigma=sigmas,
            timesteps=timesteps,
            positions=positions,
            context=prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Loss mask: only compute loss on non-conditioning target tokens
        # Reference tokens: all False (no loss) across every reference
        # Target tokens: True where not conditioning
        ref_loss_mask = torch.zeros(batch_size, total_ref_seq_len, dtype=torch.bool, device=device)
        target_loss_mask = ~target_conditioning_mask
        video_loss_mask = torch.cat([ref_loss_mask, target_loss_mask], dim=1)

        return ModelInputs(
            video=video_modality,
            audio=None,
            video_targets=targets,
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
            ref_seq_len=total_ref_seq_len,
        )

    def compute_loss(
        self,
        video_pred: Tensor,
        _audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute masked loss only on target portion. Returns [B,]."""
        # Extract target portion of prediction
        ref_seq_len = inputs.ref_seq_len
        target_pred = video_pred[:, ref_seq_len:, :]

        # Get target portion of loss mask
        target_loss_mask = inputs.video_loss_mask[:, ref_seq_len:]

        # Compute per-element loss [B,]
        loss = (target_pred - inputs.video_targets).pow(2)
        loss_mask = target_loss_mask.unsqueeze(-1).float()
        masked = loss.mul(loss_mask)
        return masked.mean(dim=[-2, -1]) / loss_mask.mean(dim=[-2, -1]).clamp(min=1e-8)

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Get metadata for checkpoint files."""
        metadata: dict[str, Any] = {}
        # Always include reference_downscale_factor for IC-LoRAs so inference
        # pipelines know the expected scale factor for reference videos.
        if self.reference_downscale_factor is not None:
            metadata["reference_downscale_factor"] = self.reference_downscale_factor
        return metadata

    @staticmethod
    def _infer_reference_downscale_factor(
        target_height: int,
        target_width: int,
        ref_height: int,
        ref_width: int,
    ) -> int:
        """Infer the reference downscale factor from target and reference dimensions."""
        # If dimensions match, no scaling needed
        if target_height == ref_height and target_width == ref_width:
            return 1

        # Calculate scale factors for each dimension
        if target_height % ref_height != 0 or target_width % ref_width != 0:
            raise ValueError(
                f"Target dimensions ({target_height}x{target_width}) must be exact multiples "
                f"of reference dimensions ({ref_height}x{ref_width})"
            )

        scale_h = target_height // ref_height
        scale_w = target_width // ref_width

        if scale_h != scale_w:
            raise ValueError(
                f"Reference scale must be uniform. Got height scale {scale_h} and width scale {scale_w}. "
                f"Target: {target_height}x{target_width}, Reference: {ref_height}x{ref_width}"
            )

        if scale_h < 1:
            raise ValueError(
                f"Reference dimensions ({ref_height}x{ref_width}) cannot be larger than "
                f"target dimensions ({target_height}x{target_width})"
            )

        return scale_h
