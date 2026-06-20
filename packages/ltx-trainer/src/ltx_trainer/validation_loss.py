"""Deterministic validation-loss evaluation.

Computes a held-out validation loss on a fixed sigma grid with seeded noise, so
the value reflects model quality rather than timestep/noise sampling variance.
Useful for overfitting detection and comparing runs (diffusion val loss
correlates only weakly with sample quality, so use it to complement, not replace,
the visual val samples).

Determinism is achieved without modifying the training strategy:
  - the fixed sigma grid is supplied via a ``FixedTimestepSampler`` passed to the
    strategy's normal ``prepare_training_inputs(batch, sampler)`` entry point;
  - reproducible noise comes from seeding the global RNG per (batch, timestep),
    with the RNG state saved and restored around the whole run so training
    determinism is unaffected.
"""

from collections.abc import Callable

import torch
from accelerate import Accelerator, DistributedType
from torch import Tensor
from torch.utils.data import DataLoader

from ltx_trainer import logger
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.datasets import PrecomputedDataset
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import TrainingStrategy


class FixedTimestepSampler(TimestepSampler):
    """Timestep sampler that returns a constant sigma for every batch element.

    Lets the val-loss path evaluate the model at a fixed point on the noise
    schedule (no sampling variance) while reusing the strategy's normal
    ``prepare_training_inputs(batch, sampler)`` entry point — no strategy changes.
    """

    def __init__(self, sigma: float) -> None:
        self._sigma = float(sigma)

    def sample(self, batch_size: int, seq_length: int | None = None, device: torch.device = None) -> Tensor:  # noqa: ARG002
        return torch.full((batch_size,), self._sigma, device=device)

    def sample_for(self, batch: Tensor) -> Tensor:
        return torch.full((batch.shape[0],), self._sigma, device=batch.device, dtype=batch.dtype)


class ValidationLossEvaluator:
    """Compute and aggregate a deterministic validation loss over a held-out set.

    Owns its own val DataLoader (built lazily from ``validation.val_data_root``)
    and reuses the trainer's strategy, transformer, and embedding-connector step.
    Returns a metrics dict; the trainer is responsible for logging it so all W&B
    logging shares one explicit step axis.
    """

    def __init__(
        self,
        *,
        config: LtxTrainerConfig,
        strategy: TrainingStrategy,
        transformer: torch.nn.Module,
        accelerator: Accelerator,
        apply_connectors: Callable[[dict[str, dict[str, Tensor]]], None],
    ) -> None:
        self._config = config
        self._strategy = strategy
        self._transformer = transformer
        self._accelerator = accelerator
        self._apply_connectors = apply_connectors
        self._dataloader: DataLoader | None = None

    def _ensure_dataloader(self) -> None:
        """Build the val DataLoader once. shuffle=False, drop_last=False for determinism."""
        if self._dataloader is not None:
            return
        val_cfg = self._config.validation
        data_sources = self._strategy.get_data_sources()
        dataset = PrecomputedDataset(val_cfg.val_data_root, data_sources=data_sources)
        logger.info(f"Loaded val dataset with {len(dataset):,} samples from {val_cfg.val_data_root}")

        num_workers = self._config.data.num_dataloader_workers
        loader = DataLoader(
            dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
        )
        self._dataloader = self._accelerator.prepare(loader)

    @staticmethod
    def _batch_size(batch: dict[str, dict[str, Tensor]]) -> int:
        for key in ("video_latents", "audio_latents"):
            if key in batch:
                return batch[key]["latents"].shape[0]
        raise KeyError("batch has neither 'video_latents' nor 'audio_latents'")

    @torch.inference_mode()
    def run(self, global_step: int) -> dict[str, float] | None:
        """Run the val-loss pass and return aggregated metrics (or None if skipped).

        Distributed: under FSDP all ranks participate; under DDP only the main
        process runs (mirrors the trainer's validation-sampling convention).
        """
        if self._accelerator.distributed_type != DistributedType.FSDP and not self._accelerator.is_main_process:
            return None

        self._ensure_dataloader()
        val_cfg = self._config.validation
        timesteps = list(val_cfg.val_loss_timesteps)
        max_samples = val_cfg.val_loss_max_samples

        per_timestep_sums: dict[float, float] = dict.fromkeys(timesteps, 0.0)
        per_timestep_counts: dict[float, int] = dict.fromkeys(timesteps, 0)

        was_training = self._transformer.training
        self._transformer.eval()

        # Save/restore RNG so the seeded val noise does not perturb training determinism.
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            samples_seen = 0
            for batch_idx, batch in enumerate(self._dataloader):
                if max_samples is not None and samples_seen >= max_samples:
                    break
                samples_seen += self._batch_size(batch)

                # Connectors convert precomputed features to context embeds in place; apply once
                # per batch (sigma-independent), then evaluate the loss at each fixed timestep.
                self._apply_connectors(batch)

                for t_idx, sigma in enumerate(timesteps):
                    # Deterministic per (run, batch, timestep): same val data + same model -> same
                    # loss; different across (batch, timestep) so noise patterns are not reused.
                    torch.manual_seed(val_cfg.val_loss_seed + batch_idx * len(timesteps) + t_idx)
                    loss = self._forward_loss(batch, sigma)
                    per_timestep_sums[sigma] += float(loss.sum().item())
                    per_timestep_counts[sigma] += int(loss.numel())
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            if was_training:
                self._transformer.train()

        return self._aggregate(timesteps, per_timestep_sums, per_timestep_counts, global_step)

    def _forward_loss(self, batch: dict[str, dict[str, Tensor]], sigma: float) -> Tensor:
        """Strategy forward + per-element loss at a fixed sigma. Returns loss [B,]."""
        model_inputs = self._strategy.prepare_training_inputs(batch, FixedTimestepSampler(sigma))
        video_pred, audio_pred = self._transformer(
            video=model_inputs.video,
            audio=model_inputs.audio,
            perturbations=None,
        )
        return self._strategy.compute_loss(video_pred, audio_pred, model_inputs)

    def _aggregate(
        self,
        timesteps: list[float],
        sums: dict[float, float],
        counts: dict[float, int],
        global_step: int,
    ) -> dict[str, float] | None:
        """Aggregate per-timestep losses into overall mean, per-timestep, and sigma buckets."""
        metrics: dict[str, float] = {}
        timestep_means: list[float] = []
        for t in timesteps:
            if counts[t] == 0:
                continue
            mean = sums[t] / counts[t]
            # Key format: val/loss_t05, val/loss_t95 — sortable in W&B charts
            metrics[f"val/loss_t{round(t * 100):02d}"] = mean
            timestep_means.append(mean)

        if not timestep_means:
            logger.warning("Val loss requested but no val samples were processed; check val_data_root.")
            return None

        metrics["val/loss"] = sum(timestep_means) / len(timestep_means)

        # Sigma buckets for a quick "low/high noise" trend reading.
        low = [m for t, m in zip(timesteps, timestep_means, strict=True) if t < 0.3]
        high = [m for t, m in zip(timesteps, timestep_means, strict=True) if t >= 0.7]
        if low:
            metrics["val/loss_low_sigma"] = sum(low) / len(low)
        if high:
            metrics["val/loss_high_sigma"] = sum(high) / len(high)

        logger.info(
            f"Val loss @ step {global_step}: mean={metrics['val/loss']:.4f}"
            + (f" low={metrics['val/loss_low_sigma']:.4f}" if "val/loss_low_sigma" in metrics else "")
            + (f" high={metrics['val/loss_high_sigma']:.4f}" if "val/loss_high_sigma" in metrics else "")
        )
        return metrics
