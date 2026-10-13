"""Cached model loading for Hugging Face, timm and torchvision (all optional)."""
from __future__ import annotations

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
