"""Zero-code testing: turn a model spec into a model factory and example inputs.

    pytest --torch-model=hf:bert-base-uncased
    pytest --torch-model=timm:resnet50 --torch-model=torchvision:mobilenet_v3_small
    pytest --torch-model=mypackage.models:Classifier
    pytest --torch-model=path/to/models.py:build_model --torch-model-inputs=4x3x32x32

Weights are random by default (Hugging Face models only download their config);
``--torch-model-pretrained`` loads pretrained weights instead.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

BATCH = 4
SEQ_LEN = 16
IMAGE_SIZE = 64


class SpecError(ValueError):
    """A model spec that can't be resolved, with a message saying how to fix it."""


@dataclass
class InputPlan:
    """How to make example inputs for a model."""

    make: Callable[[torch.Generator], Any]
    kind: str = "generic"  # "text" (ids + mask), "vision" (images), "generic"
    description: str = ""
    vocab_size: int | None = None
    ids_key: str = "input_ids"
    mask_key: str = "attention_mask"


@dataclass
class ModelUnderTest:
    spec: str
    build: Callable[[], nn.Module]  # a fresh model in eval mode, on the CPU
    inputs: InputPlan
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# input plans
# --------------------------------------------------------------------------- #
def _normal(shape: tuple[int, ...]) -> Callable[[torch.Generator], torch.Tensor]:
    return lambda g: torch.randn(shape, generator=g)


def _ids(shape: tuple[int, ...], low: int, high: int) -> Callable[[torch.Generator], torch.Tensor]:
    return lambda g: torch.randint(low, max(high, low + 1), shape, generator=g)


def padded_mask(batch: int, length: int) -> torch.Tensor:
    """Ones, with real padding on some rows: row 1 loses a quarter, row 2 half."""
    mask = torch.ones(batch, length, dtype=torch.long)
    if batch > 1:
        mask[1, length - max(1, length // 4):] = 0
    if batch > 2:
        mask[2, length // 2:] = 0
    return mask


def text_plan(vocab_size: int, length: int, *, extra: dict | None = None, note: str = "") -> InputPlan:
    low = 1 if vocab_size > 1 else 0  # keep id 0 (often padding) out of the real tokens

    def make(g: torch.Generator) -> dict:
        inputs = {"input_ids": torch.randint(low, vocab_size, (BATCH, length), generator=g),
                  "attention_mask": padded_mask(BATCH, length)}
        for key, factory in (extra or {}).items():
            inputs[key] = factory(g)
        return inputs

    keys = ", ".join(["attention_mask (padded)", *(extra or {})])
    return InputPlan(make, "text", f"input_ids long ({BATCH}, {length}) in [{low}, {vocab_size}), {keys}{note}",
                     vocab_size=vocab_size)


def vision_plan(channels: int, height: int, width: int, key: str | None = None) -> InputPlan:
    shape = (BATCH, channels, height, width)
    tensor = _normal(shape)
    make = (lambda g: {key: tensor(g)}) if key else tensor
    return InputPlan(make, "vision", f"{key or 'images'} float {shape}")


_DTYPES = {"float": torch.float32, "float32": torch.float32, "float16": torch.float16,
           "bfloat16": torch.bfloat16, "float64": torch.float64, "long": torch.long,
           "int": torch.int32, "bool": torch.bool, "mask": None}


def _mask_factory(shape: tuple[int, ...]) -> Callable[[torch.Generator], torch.Tensor]:
    def make(_g: torch.Generator) -> torch.Tensor:
        return padded_mask(*shape)

    return make


def _random_factory(shape: tuple[int, ...], dtype: torch.dtype, high: int) -> Callable[[torch.Generator], torch.Tensor]:
    def make(g: torch.Generator) -> torch.Tensor:
        if dtype.is_floating_point:
            return torch.randn(shape, generator=g).to(dtype)
        return torch.randint(0, high, shape, generator=g).to(dtype)

    return make


def parse_input_spec(text: str) -> InputPlan:
    """``--torch-model-inputs``: comma-separated tensors, each ``[name=][dtype:]AxBxC[:high]``.

    ``4x16`` (float normal), ``long:4x16:1000`` (ids in [0, 1000)), ``mask:4x16`` (an
    attention mask with padding). Named entries become keyword arguments.
    """
    named: dict[str, Callable[[torch.Generator], torch.Tensor]] = {}
    positional: list[Callable[[torch.Generator], torch.Tensor]] = []
    factory: Callable[[torch.Generator], torch.Tensor]
    make: Callable[[torch.Generator], Any]
    descriptions = []
    for item in [p.strip() for p in text.split(",") if p.strip()]:
        name, _, body = item.rpartition("=")
        parts = body.split(":")
        dtype_name = parts[0] if parts[0] in _DTYPES else "float"
        rest = parts[1:] if parts[0] in _DTYPES else parts
        if not rest or not re.fullmatch(r"\d+(x\d+)*", rest[0]):
            raise SpecError(f"--torch-model-inputs: can't read {item!r}; expected e.g. 4x16, long:4x16:1000, mask:4x16")
        shape = tuple(int(n) for n in rest[0].split("x"))
        high = int(rest[1]) if len(rest) > 1 else 100
        if dtype_name == "mask":
            if len(shape) != 2:
                raise SpecError(f"--torch-model-inputs: a mask is (batch, length), got {shape}")
            factory = _mask_factory(shape)
        else:
            dtype = _DTYPES[dtype_name]
            assert dtype is not None  # only "mask" has no dtype
            factory = _random_factory(shape, dtype, high)
        (named.__setitem__(name, factory) if name else positional.append(factory))
        descriptions.append(item)
    if named and positional:
        raise SpecError("--torch-model-inputs: use either all named (key=...) or all positional inputs")
    if named:
        make = lambda g: {k: f(g) for k, f in named.items()}  # noqa: E731
        is_text = "input_ids" in named and "attention_mask" in named
    else:
        make = (lambda g: positional[0](g)) if len(positional) == 1 else (lambda g: tuple(f(g) for f in positional))
        is_text = False
    vocab = None
    if is_text:
        match = re.search(r"input_ids=\w+:\d+(?:x\d+)*:(\d+)", text)
        vocab = int(match.group(1)) if match else 100
    return InputPlan(make, "text" if is_text else "generic", "from --torch-model-inputs: " + ", ".join(descriptions),
                     vocab_size=vocab)


def infer_from_layers(model: nn.Module) -> InputPlan:
    """Guess inputs from the first layer that takes them (Linear, Conv, Embedding, RNNs)."""
    shape: tuple[int, ...]
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            vocab = module.num_embeddings
            return InputPlan(_ids((BATCH, SEQ_LEN), 0, vocab), "generic",
                             f"ids long ({BATCH}, {SEQ_LEN}) in [0, {vocab}) (from the first Embedding)")
        if isinstance(module, nn.Linear):
            shape = (BATCH, module.in_features)
            return InputPlan(_normal(shape), "generic", f"float {shape} (from the first Linear)")
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            spatial: tuple[int, ...] = {nn.Conv1d: (IMAGE_SIZE,), nn.Conv2d: (IMAGE_SIZE, IMAGE_SIZE),
                                        nn.Conv3d: (8, 32, 32)}[type(module)]
            shape = (BATCH, module.in_channels, *spatial)
            kind = "vision" if isinstance(module, nn.Conv2d) else "generic"
            return InputPlan(_normal(shape), kind, f"float {shape} (from the first {type(module).__name__})")
        if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
            if not module.batch_first:
                break  # sequence-first: batch dim unknown to us
            shape = (BATCH, SEQ_LEN, module.input_size)
            return InputPlan(_normal(shape), "generic", f"float {shape} (from the first {type(module).__name__})")
        if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerEncoder)):
            layer = module.layers[0] if isinstance(module, nn.TransformerEncoder) else module
            if not layer.self_attn.batch_first:
                break
            shape = (BATCH, SEQ_LEN, layer.self_attn.embed_dim)
            return InputPlan(_normal(shape), "generic", f"float {shape} (from the first transformer layer)")
    raise SpecError(
        f"can't infer inputs for {type(model).__name__}. Give the model an "
        "`example_inputs(self, batch_size)` method, or pass --torch-model-inputs (e.g. 4x3x32x32 or "
        "input_ids=long:4x16:1000,attention_mask=mask:4x16)"
    )


def example_inputs_plan(model: nn.Module) -> InputPlan | None:
    provider = getattr(model, "example_inputs", None)
    if provider is None:
        return None
    if callable(provider):
        params = inspect.signature(provider).parameters
        make = (lambda _g: provider(BATCH)) if params else (lambda _g: provider())
        return InputPlan(make, "generic", "from the model's example_inputs()")
    return InputPlan(lambda _g: provider, "generic", "the model's example_inputs attribute")


# --------------------------------------------------------------------------- #
# resolvers
# --------------------------------------------------------------------------- #
def _need(module: str, extra: str):
    try:
        return importlib.import_module(module)
    except ImportError:
        raise SpecError(f"{module} is not installed: pip install 'pytest-torch[{extra}]'") from None


def _hf(name: str, pretrained: bool, inputs_override: InputPlan | None) -> ModelUnderTest:
    transformers = _need("transformers", "hf")
    name, _, auto_class = name.partition("@")
    auto = getattr(transformers, auto_class or "AutoModel", None)
    if auto is None:
        raise SpecError(f"transformers has no {auto_class!r}")
    try:
        config = transformers.AutoConfig.from_pretrained(name)
    except Exception as exc:  # offline, unknown name, private repo, ...
        raise SpecError(f"can't load the Hugging Face config for {name!r}: {str(exc).splitlines()[0]}") from None

    def build() -> nn.Module:
        model = auto.from_pretrained(name) if pretrained else auto.from_config(config)
        return model.eval()

    if inputs_override is not None:
        return ModelUnderTest(f"hf:{name}", build, inputs_override)
    params = inspect.signature(build().forward).parameters
    vocab = getattr(config, "vocab_size", None)
    if "input_ids" in params and vocab:
        length = min(SEQ_LEN, getattr(config, "max_position_embeddings", SEQ_LEN) or SEQ_LEN)
        extra: dict[str, Callable] = {}
        if getattr(config, "is_encoder_decoder", False) and "decoder_input_ids" in params:
            # A different length than the encoder's, so the two can't be mistaken as aligned.
            extra["decoder_input_ids"] = _ids((BATCH, max(2, length // 2)), 1, vocab)
        if "use_cache" in params:
            # The attention cache isn't the model's answer, and its length follows the input.
            extra["use_cache"] = lambda _g: False
        return ModelUnderTest(f"hf:{name}", build, text_plan(vocab, length, extra=extra))
    if "pixel_values" in params:
        size = getattr(config, "image_size", 224)
        height, width = (size, size) if isinstance(size, int) else tuple(size)
        return ModelUnderTest(f"hf:{name}", build, vision_plan(getattr(config, "num_channels", 3), height, width, "pixel_values"))
    raise SpecError(
        f"can't infer inputs for {name!r} (forward takes {list(params)[:6]}); pass --torch-model-inputs"
    )


def _timm(name: str, pretrained: bool, inputs_override: InputPlan | None) -> ModelUnderTest:
    timm = _need("timm", "timm")
    if name not in timm.list_models() and not pretrained:
        raise SpecError(f"timm has no model {name!r}")

    def build() -> nn.Module:
        return timm.create_model(name, pretrained=pretrained).eval()

    if inputs_override is not None:
        return ModelUnderTest(f"timm:{name}", build, inputs_override)
    cfg = getattr(build(), "pretrained_cfg", None) or {}
    channels, height, width = cfg.get("input_size", (3, 224, 224))
    return ModelUnderTest(f"timm:{name}", build, vision_plan(channels, height, width))


def _torchvision(name: str, pretrained: bool, inputs_override: InputPlan | None) -> ModelUnderTest:
    torchvision = _need("torchvision", "torchvision")
    models = torchvision.models
    available = set(models.list_models(module=models)) | set(models.list_models(module=models.segmentation))
    if name not in available:
        raise SpecError(f"{name!r} is not a torchvision classification or segmentation model")

    def build() -> nn.Module:
        return models.get_model(name, weights="DEFAULT" if pretrained else None).eval()

    if inputs_override is not None:
        return ModelUnderTest(f"torchvision:{name}", build, inputs_override)
    size = 299 if name.startswith("inception") else 224
    return ModelUnderTest(f"torchvision:{name}", build, vision_plan(3, size, size))


def _import_target(target: str) -> Any:
    module_part, _, attr = target.rpartition(":")
    if not module_part or not attr:
        raise SpecError(f"expected package.module:Factory or path/to/file.py:Factory, got {target!r}")
    if module_part.endswith(".py"):
        path = Path(module_part).resolve()
        if not path.is_file():
            raise SpecError(f"no such file: {module_part}")
        spec = importlib.util.spec_from_file_location(f"_torch_model_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise SpecError(f"can't import {module_part} as a Python file")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(module_part)
        except ImportError as exc:
            raise SpecError(f"can't import {module_part!r}: {exc}") from None
    try:
        return getattr(module, attr)
    except AttributeError:
        raise SpecError(f"{module_part!r} has no attribute {attr!r}") from None


def _local(target: str, inputs_override: InputPlan | None) -> ModelUnderTest:
    factory = _import_target(target)

    def build() -> nn.Module:
        model = factory() if callable(factory) and not isinstance(factory, nn.Module) else factory
        if not isinstance(model, nn.Module):
            raise SpecError(f"{target} produced a {type(model).__name__}, not an nn.Module")
        return model.eval()

    model = build()
    plan = inputs_override or example_inputs_plan(model) or infer_from_layers(model)
    if plan.kind == "generic" and isinstance(plan.make(torch.Generator()), dict):
        sample = plan.make(torch.Generator())
        if "input_ids" in sample and "attention_mask" in sample:
            plan.kind = "text"
    return ModelUnderTest(target, build, plan)


def resolve(spec: str, *, pretrained: bool = False, inputs: str | None = None) -> ModelUnderTest:
    """Resolve a ``--torch-model`` spec; raises :class:`SpecError` with a fix on failure."""
    override = parse_input_spec(inputs) if inputs else None
    scheme, _, rest = spec.partition(":")
    if scheme == "hf" and rest:
        subject = _hf(rest, pretrained, override)
    elif scheme == "timm" and rest:
        subject = _timm(rest, pretrained, override)
    elif scheme == "torchvision" and rest:
        subject = _torchvision(rest, pretrained, override)
    else:
        subject = _local(spec, override)
    subject.spec = spec
    # Fail early and clearly if the inferred inputs don't fit the model.
    model = subject.build()
    sample = subject.inputs.make(torch.Generator().manual_seed(0))
    try:
        with torch.inference_mode():
            from .utils import call_model

            call_model(model, sample)
    except Exception as exc:
        raise SpecError(
            f"the inferred inputs ({subject.inputs.description}) don't fit the model: "
            f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}. Pass --torch-model-inputs."
        ) from None
    return subject
