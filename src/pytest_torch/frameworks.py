"""Lightning and Hugging Face ``Trainer`` support: trainers set up for tests, and an adapter
that runs a ``LightningModule`` under the training-loop checks. Both packages are optional."""
from __future__ import annotations

import inspect
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from .training_loop import StepFn, TrainingSetup


# --------------------------------------------------------------------------- #
# Lightning
# --------------------------------------------------------------------------- #
def _lightning() -> Any:
    try:
        import lightning.pytorch as pl
    except ImportError:
        try:
            import pytorch_lightning as pl
        except ImportError:
            raise ImportError("this needs Lightning: pip install 'pytest-torch[lightning]'") from None
    return pl


def lightning_trainer_kwargs(device: Any, root_dir: str | Path) -> dict[str, Any]:
    """``Trainer`` arguments for a short, quiet run on ``device``: 5 steps, no logger,
    checkpoints, progress bar, model summary or sanity validation, files under
    ``root_dir``."""
    kind, _, index = str(device).partition(":")
    if kind == "cuda":
        accelerator, devices = "gpu", [int(index)] if index else 1
    elif kind in ("cpu", "mps"):
        accelerator, devices = kind, 1
    else:
        pytest.skip(f"Lightning has no accelerator for {device}")
    return {
        "accelerator": accelerator,
        "devices": devices,
        "max_steps": 5,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
        "log_every_n_steps": 1,
        "default_root_dir": str(root_dir),
    }


def _schedulers(config: Any) -> tuple[list[Any], list[dict[str, Any]]]:
    """Optimizers and scheduler configs from ``configure_optimizers()``'s many forms."""
    def scheduler(entry: Any) -> dict[str, Any]:
        if isinstance(entry, Mapping):
            return {"scheduler": entry["scheduler"], "interval": entry.get("interval", "epoch"),
                    "frequency": entry.get("frequency", 1)}
        return {"scheduler": entry, "interval": "epoch", "frequency": 1}

    if isinstance(config, torch.optim.Optimizer):
        return [config], []
    if isinstance(config, Mapping):
        found = [scheduler(config["lr_scheduler"])] if config.get("lr_scheduler") is not None else []
        return [config["optimizer"]], found
    if isinstance(config, (list, tuple)):
        if len(config) == 2 and all(isinstance(part, (list, tuple)) for part in config):
            return list(config[0]), [scheduler(s) for s in config[1]]
        optimizers, found = [], []
        for entry in config:
            if isinstance(entry, Mapping):
                more, scheds = _schedulers(entry)
                optimizers += more
                found += scheds
            else:
                optimizers.append(entry)
        return optimizers, found
    raise TypeError(f"configure_optimizers() returned {type(config).__name__}, which isn't supported here")


def from_lightning(
    make_module: Callable[[], Any], *, epoch_length: int | None = None
) -> tuple[Callable[[], TrainingSetup], StepFn]:
    """``(make_setup, step)`` for the training-loop checks, from a ``LightningModule``.

    ``make_setup`` builds the module and calls its ``configure_optimizers()``; ``step``
    does what Lightning's automatic optimization does for one batch: ``training_step``,
    backward, optimizer step, zero grad, and the schedulers whose ``interval`` is
    ``"step"`` (every ``frequency`` steps). There are no epochs in a step-based run: pass
    ``epoch_length`` (steps per epoch) to step ``"epoch"`` schedulers too.
    ``training_step`` runs without a ``Trainer``: ``self.log`` is ignored and
    ``self.global_step`` reads 0; steps that need ``self.trainer`` need the
    ``lightning_trainer`` fixture instead. Manual optimization isn't supported.

        make_setup, step = from_lightning(lambda: LitClassifier(lr=1e-3))
        training_runner.run(make_setup, batches, step).assert_healthy()
    """
    def make_setup() -> TrainingSetup:
        module = make_module()
        if not getattr(module, "automatic_optimization", True):
            raise TypeError("from_lightning supports automatic optimization only; "
                            "test manual optimization with the lightning_trainer fixture")
        optimizers, schedulers = _schedulers(module.configure_optimizers())
        if len(optimizers) != 1:
            raise TypeError(f"automatic optimization uses one optimizer, configure_optimizers() returned "
                            f"{len(optimizers)}")
        setup = TrainingSetup(module, optimizers, schedulers=[s["scheduler"] for s in schedulers])
        setup.lightning_schedulers = schedulers  # type: ignore[attr-defined]
        return setup

    def step(setup: TrainingSetup, batch: Any) -> Any:
        module, optimizer = setup.model, setup.optimizer
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*self.log.*")  # no Trainer to log to
            output = module.training_step(batch, setup.step_index)
        loss = output["loss"] if isinstance(output, Mapping) else output
        if loss is None:  # Lightning skips the optimizer step
            return None
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        done = setup.step_index + 1
        for config in setup.lightning_schedulers:  # type: ignore[attr-defined]
            if config["interval"] == "step":
                due = done % config["frequency"] == 0
            else:
                due = epoch_length is not None and done % (epoch_length * config["frequency"]) == 0
            if due:
                config["scheduler"].step()
        return loss.detach()

    return make_setup, step


# --------------------------------------------------------------------------- #
# Hugging Face Trainer
# --------------------------------------------------------------------------- #
def hf_training_arguments(device: Any, output_dir: str | Path, seed: int = 0, **overrides: Any) -> Any:
    """``TrainingArguments`` for a short, quiet run on ``device``: 5 steps, batch size 4,
    the loss logged every step, nothing saved or reported, ``seed`` for model init and
    data order. ``overrides`` set other arguments."""
    transformers = _transformers()
    accepted = inspect.signature(transformers.TrainingArguments).parameters
    kind = str(device).split(":")[0]
    if kind == "xpu":
        pytest.skip("the Hugging Face Trainer picks XPU devices itself: use the cuda or cpu runs")
    values: dict[str, Any] = {
        "output_dir": str(output_dir),
        "max_steps": 5,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        "seed": seed,
        "data_seed": seed,
        "disable_tqdm": True,
        "dataloader_num_workers": 0,
    }
    if kind == "cpu":
        values["use_cpu" if "use_cpu" in accepted else "no_cuda"] = True
    values.update(overrides)
    if "evaluation_strategy" in values and "eval_strategy" in accepted and "evaluation_strategy" not in accepted:
        values["eval_strategy"] = values.pop("evaluation_strategy")
    return transformers.TrainingArguments(**values)


def _transformers() -> Any:
    try:
        import transformers
    except ImportError:
        raise ImportError("this needs transformers: pip install 'pytest-torch[hf]'") from None
    return transformers


def as_dataset(data: Any) -> Any:
    """A dict of equally long tensors (e.g. from ``make_batch.tokens``) becomes a dataset of
    per-row dicts; anything else is returned unchanged."""
    if isinstance(data, Mapping) and all(isinstance(v, torch.Tensor) for v in data.values()):
        from .data import _DictDataset

        return _DictDataset({k: v.cpu() for k, v in data.items()})  # the Trainer moves batches
    return data
