"""Cached model loading for Hugging Face, timm and torchvision (all optional)."""
from __future__ import annotations

import copy
from typing import Any, Callable


def _finalize(model: Any, device: Any, dtype: Any) -> Any:
    model.eval()
    if dtype is not None:
        model = model.to(dtype)
    return model.to(device)


def _freeze(kwargs: dict) -> str:
    return repr(sorted(kwargs.items()))


def _need(package: str, extra: str) -> Callable[[], Any]:
    def importer():
        import importlib

        try:
            return importlib.import_module(package)
        except ImportError as exc:
            raise ImportError(
                f"ModelLoader needs {package}: pip install 'pytest-torch[{extra}]'"
            ) from exc

    return importer


class ModelLoader:
    """Loads models in eval mode and caches them, keyed by every argument.

    Cached instances are shared: don't mutate them (use ``copy.deepcopy`` first), and
    note that pretrained weights are downloaded on first use.
    """

    def __init__(self, device: Any = "cpu"):
        self.device = device
        self._cache: dict[tuple, Any] = {}

    def _get(self, key: tuple, factory: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    def hf(self, name: str, *, auto_class: str = "AutoModel", dtype: Any = None,
           device: Any = None, **kwargs) -> Any:
        """``transformers.<auto_class>.from_pretrained(name, **kwargs)`` (path or hub id)."""
        transformers = _need("transformers", "hf")()
        device = self.device if device is None else device

        def make():
            cls = getattr(transformers, auto_class)
            return _finalize(cls.from_pretrained(name, **kwargs), device, dtype)

        key = ("hf", name, auto_class, _freeze(kwargs), str(device), str(dtype))
        return self._get(key, make)

    def hf_tiny(self, name: Any, *, auto_class: str = "AutoModel", layers: int = 2, hidden_size: int = 64,
                heads: int = 4, vocab_size: int | None = None, seed: int = 0, dtype: Any = None,
                device: Any = None, trust_remote_code: bool = False, **overrides) -> Any:
        """A small, randomly initialized model with the architecture of ``name``.

        ``auto_class`` is an Auto class (default ``AutoModel``) or a model class name such
        as ``"LlavaForConditionalGeneration"``.

        Only the config is loaded (``name`` is a hub id, a local path or a
        ``PretrainedConfig``); :func:`shrink_config` reduces it to ``layers`` layers of width
        ``hidden_size`` with ``heads`` attention heads, then ``overrides`` set config
        attributes. Weights are initialized from ``seed``, so the model is the same in every
        run whatever the test order. ``model.config`` is the shrunk config.
        """
        transformers = _need("transformers", "hf")()
        device = self.device if device is None else device

        def make():
            import torch

            if isinstance(name, str):
                config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=trust_remote_code)
            else:
                config = copy.deepcopy(name)
            shrink_config(config, layers=layers, hidden_size=hidden_size, heads=heads, vocab_size=vocab_size,
                          **overrides)
            cls = getattr(transformers, auto_class)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                if hasattr(cls, "from_config"):  # an Auto class
                    model = cls.from_config(config, trust_remote_code=trust_remote_code)
                else:  # a model class, e.g. LlavaForConditionalGeneration
                    model = cls(config)
            return _finalize(model, device, dtype)

        source = name if isinstance(name, str) else name.to_json_string()
        key = ("hf_tiny", source, auto_class, layers, hidden_size, heads, vocab_size, seed,
               _freeze(overrides), str(device), str(dtype), trust_remote_code)
        return self._get(key, make)

    def hf_tokenizer(self, name: str, **kwargs) -> Any:
        transformers = _need("transformers", "hf")()
        key = ("hf_tokenizer", name, _freeze(kwargs))
        return self._get(key, lambda: transformers.AutoTokenizer.from_pretrained(name, **kwargs))

    def timm(self, name: str, *, pretrained: bool = False, dtype: Any = None,
             device: Any = None, **kwargs) -> Any:
        """``timm.create_model(name, pretrained=..., **kwargs)``; random init by default."""
        timm = _need("timm", "timm")()
        device = self.device if device is None else device

        def make():
            return _finalize(timm.create_model(name, pretrained=pretrained, **kwargs), device, dtype)

        key = ("timm", name, pretrained, _freeze(kwargs), str(device), str(dtype))
        return self._get(key, make)

    def torchvision(self, name: str, *, weights: Any = None, dtype: Any = None,
                    device: Any = None, **kwargs) -> Any:
        """``torchvision.models.get_model(name, weights=...)``; random init by default."""
        torchvision = _need("torchvision", "torchvision")()
        device = self.device if device is None else device

        def make():
            model = torchvision.models.get_model(name, weights=weights, **kwargs)
            return _finalize(model, device, dtype)

        key = ("torchvision", name, str(weights), _freeze(kwargs), str(device), str(dtype))
        return self._get(key, make)


# --------------------------------------------------------------------------- #
# Tiny Hugging Face configs
# --------------------------------------------------------------------------- #
# Config attributes by role; a config only has some of them (aliases such as GPT-2's
# n_layer for num_hidden_layers resolve either way).
_LAYERS = ("num_hidden_layers", "n_layer", "num_layers", "n_layers", "num_decoder_layers",
           "encoder_layers", "decoder_layers", "num_encoder_layers")
_HIDDEN = ("hidden_size", "n_embd", "d_model", "dim", "embed_dim", "word_embed_proj_dim")
_HEADS = ("num_attention_heads", "n_head", "num_heads", "n_heads", "encoder_attention_heads",
          "decoder_attention_heads")
_KV_HEADS = ("num_key_value_heads", "num_kv_heads", "multi_query_group_num")
_HEAD_DIM = ("head_dim", "d_kv", "kv_channels")
_INTERMEDIATE = ("intermediate_size", "n_inner", "d_ff", "ffn_dim", "encoder_ffn_dim", "decoder_ffn_dim",
                 "hidden_dim", "moe_intermediate_size", "shared_expert_intermediate_size")
_EXPERTS = ("num_local_experts", "num_experts", "n_routed_experts")
_EXPERTS_PER_TOKEN = ("num_experts_per_tok", "moe_topk")
_AT_MOST_LAYERS = ("max_window_layers", "first_k_dense_replace")


def _present(config: Any, keys: tuple) -> list[str]:
    stored = vars(config)
    aliases = getattr(config, "attribute_map", {})
    found = []
    for key in keys:
        target = aliases.get(key, key)
        if target in stored and stored[target] is not None and target not in found:
            found.append(target)
    return found


def shrink_config(config: Any, *, layers: int = 2, hidden_size: int = 64, heads: int = 4,
                  vocab_size: int | None = None, **overrides) -> Any:
    """Shrink a Hugging Face ``PretrainedConfig`` in place (and return it) for fast tests.

    Sets the layer counts, hidden size, attention heads (key/value heads too, keeping
    grouped-query attention grouped), head dimension, feed-forward sizes and expert counts
    under every name architectures use for them, truncates per-layer lists (such as
    ``layer_types``) and recurses into nested configs (``text_config``, ``vision_config``,
    ``encoder``, ``decoder``). The vocabulary is kept unless ``vocab_size`` is given, so a
    real tokenizer's ids stay valid. ``overrides`` are set last.
    """
    if hidden_size % heads:
        raise ValueError(f"hidden_size ({hidden_size}) must be divisible by heads ({heads})")
    stored = vars(config)
    old_layers = {key: stored[key] for key in _present(config, _LAYERS) if isinstance(stored[key], int)}
    for key, value in old_layers.items():
        setattr(config, key, min(value, layers))
    previous = max(old_layers.values(), default=0)
    if previous > layers:
        for key, value in list(stored.items()):
            if isinstance(value, (list, tuple)) and len(value) == previous and not key.startswith("_"):
                setattr(config, key, type(value)(value[:layers]))
    for key in _present(config, _AT_MOST_LAYERS):
        if isinstance(stored[key], int):
            setattr(config, key, min(stored[key], layers))

    old_heads = max((stored[k] for k in _present(config, _HEADS) if isinstance(stored[k], int)), default=None)
    for key in _present(config, _HIDDEN):
        if isinstance(stored[key], int):
            setattr(config, key, hidden_size)
    if "embedding_size" in stored and isinstance(stored["embedding_size"], int):  # ALBERT
        config.embedding_size = min(stored["embedding_size"], hidden_size)
    for key in _present(config, _HEADS):
        setattr(config, key, heads)
    for key in _present(config, _KV_HEADS):
        grouped = old_heads is not None and isinstance(stored[key], int) and stored[key] < old_heads
        setattr(config, key, max(1, heads // 2) if grouped else heads)
    for key in _present(config, _HEAD_DIM):
        setattr(config, key, hidden_size // heads)
    for key in _present(config, _INTERMEDIATE):
        setattr(config, key, 4 * hidden_size if key in ("intermediate_size", "n_inner", "d_ff", "ffn_dim",
                                                         "encoder_ffn_dim", "decoder_ffn_dim", "hidden_dim")
                else 2 * hidden_size)
    for key in _present(config, _EXPERTS):
        setattr(config, key, min(stored[key], 4))
    for key in _present(config, _EXPERTS_PER_TOKEN):
        setattr(config, key, min(stored[key], 2))
    if vocab_size is not None and "vocab_size" in stored:
        config.vocab_size = vocab_size

    for key, value in list(stored.items()):  # nested configs: CLIP, LLaVA, encoder-decoder
        if key.startswith("_") or not hasattr(value, "to_dict") or not hasattr(value, "attribute_map"):
            continue
        shrink_config(value, layers=layers, hidden_size=hidden_size, heads=heads, vocab_size=vocab_size)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config
