"""Default tolerances: built-in, from ``pyproject.toml``, and per-test overrides.

    [tool.pytest_torch.tolerance]
    bfloat16 = { rtol = 2e-2, atol = 1e-2 }        # any comparison of bf16 tensors
    "cuda.float32" = { rtol = 1e-4 }                # only when testing on CUDA (TF32)
    compile_parity = { rtol = 1e-3, atol = 1e-3 }   # default of assert_compile_parity

A key is a dtype name, a check name (see ``CHECK_DEFAULTS``), or either prefixed with a
device type (``cuda.``, ``xpu.``, ``mps.``, ``cpu.``). An entry may set only ``rtol`` or
only ``atol``; the other comes from the next layer. Resolution, per value:

    explicit argument > @pytest.mark.torch_tolerance > pyproject.toml > built-in default

and within a layer the device-specific key beats the generic one. The device is the test's
``torch_device`` when the test has one, else the device of the tensors or model compared.

This module must stay torch-free at import: the plugin parses the config at startup.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

# (rtol, atol) per dtype, mirroring torch.testing defaults.
DTYPE_DEFAULTS: dict[str, tuple[float, float]] = {
    "float64": (1e-7, 1e-7),
    "float32": (1.3e-6, 1e-5),
    "float16": (1e-3, 1e-5),
    "bfloat16": (1.6e-2, 1e-5),
    "complex128": (1e-7, 1e-7),
    "complex64": (1.3e-6, 1e-5),
}
#: used for dtypes without an entry (integers compare exactly anyway)
FALLBACK = (1.3e-6, 1e-5)

# (rtol, atol) defaults of the assertions that compare against a different implementation.
CHECK_DEFAULTS: dict[str, tuple[float, float]] = {
    "compile_parity": (1e-4, 1e-4),
    "export_parity": (1e-5, 1e-5),
    "aoti_parity": (1e-4, 1e-4),
    "torchscript_parity": (1e-5, 1e-5),
    "onnx_parity": (1e-3, 1e-4),
    "autocast_parity": (2e-2, 2e-2),
    "device_parity": (1e-4, 1e-4),  # check_inference_contract
    "dtype_parity": (2e-2, 2e-2),  # check_inference_contract
    "divergence": (1e-4, 1e-4),  # localize_divergence
    "gradcheck": (1e-3, 1e-5),
    "checkpoint": (0.0, 0.0),  # same weights, same device: outputs must be identical
    "layout_invariance": (1e-5, 1e-5),
}
DEVICES = ("cpu", "cuda", "xpu", "mps")
_KNOWN_DTYPES = {*DTYPE_DEFAULTS, "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"}

Entry = dict  # {"rtol": float, "atol": float}, either may be missing

_config: dict[str, Entry] = {}
_overrides: list[dict[str, Entry]] = []  # innermost last
_test_device: str | None = None


# --------------------------------------------------------------------------- #
# parsing and validation
# --------------------------------------------------------------------------- #
def _split(key: str) -> tuple[str | None, str]:
    device, _, name = key.rpartition(".")
    return (device or None), name


def parse(raw: Mapping[str, Any], source: str) -> dict[str, Entry]:
    """Validate ``{key: {rtol, atol}}`` (or ``(rtol, atol)``); raise ``ValueError`` with ``source``."""
    parsed: dict[str, Entry] = {}
    for key, value in raw.items():
        device, name = _split(str(key))
        if device is not None and device not in DEVICES:
            raise ValueError(f"{source}: unknown device {device!r} in {key!r}; use one of {list(DEVICES)}")
        if name not in _KNOWN_DTYPES and name not in CHECK_DEFAULTS:
            raise ValueError(
                f"{source}: unknown tolerance key {key!r}. Use a dtype "
                f"({', '.join(sorted(DTYPE_DEFAULTS))}), a check ({', '.join(CHECK_DEFAULTS)}), "
                "or either prefixed with a device, e.g. 'cuda.float32'"
            )
        if isinstance(value, (tuple, list)) and len(value) == 2:
            value = {"rtol": value[0], "atol": value[1]}
        if not isinstance(value, Mapping) or not value or set(value) - {"rtol", "atol"}:
            raise ValueError(f"{source}: {key!r} must be {{rtol = ..., atol = ...}} (either may be omitted)")
        entry = {}
        for field_name, number in value.items():
            if isinstance(number, bool) or not isinstance(number, (int, float)) or number < 0:
                raise ValueError(f"{source}: {key}.{field_name} must be a number >= 0, got {number!r}")
            entry[field_name] = float(number)
        parsed[str(key)] = entry
    return parsed


def load_pyproject(path) -> dict[str, Entry]:
    """Read ``[tool.pytest_torch.tolerance]`` from a ``pyproject.toml`` (empty if absent)."""
    try:
        import tomllib
    except ImportError:  # Python < 3.11
        import tomli as tomllib

    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)
    raw = data.get("tool", {}).get("pytest_torch", {}).get("tolerance", {})
    return parse(raw, f"{path} [tool.pytest_torch.tolerance]")


# --------------------------------------------------------------------------- #
# state (set by the plugin)
# --------------------------------------------------------------------------- #
def configure(entries: Mapping[str, Entry]) -> None:
    """Replace the project-wide layer (what ``pyproject.toml`` provides)."""
    _config.clear()
    _config.update(entries)


@contextlib.contextmanager
def tolerance_overrides(**entries: Any) -> Iterator[None]:
    """Temporarily override default tolerances, e.g.
    ``with tolerance_overrides(bfloat16=(3e-2, 1e-2), compile_parity={"atol": 1e-3}): ...``.

    Keys as in ``pyproject.toml``; use ``**{"cuda.float32": (1e-4, 1e-5)}`` for device keys.
    This is what ``@pytest.mark.torch_tolerance`` does for one test, class or module.
    """
    _overrides.append(parse(entries, "tolerance_overrides"))
    try:
        yield
    finally:
        _overrides.pop()


@contextlib.contextmanager
def active_test_device(device: str | None) -> Iterator[None]:
    """Device whose ``<device>.<key>`` entries apply (set by the ``torch_device`` fixture)."""
    global _test_device
    previous, _test_device = _test_device, device
    try:
        yield
    finally:
        _test_device = previous


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
def _device_type(device: Any) -> str | None:
    if _test_device is not None:
        device = _test_device
    if device is None:
        return None
    return str(getattr(device, "type", device)).split(":")[0]


def _lookup(name: str, device: Any, field_name: str) -> float | None:
    kind = _device_type(device)
    keys = ([f"{kind}.{name}"] if kind else []) + [name]
    for layer in (*reversed(_overrides), _config):
        for key in keys:
            entry = layer.get(key)
            if entry and field_name in entry:
                return entry[field_name]
    return None


def _resolve(name: str, rtol, atol, device, default: tuple[float, float]) -> tuple[float, float]:
    if rtol is None:
        rtol = _lookup(name, device, "rtol")
    if atol is None:
        atol = _lookup(name, device, "atol")
    return (default[0] if rtol is None else rtol, default[1] if atol is None else atol)


def for_dtype(dtype: Any, device: Any = None, rtol=None, atol=None) -> tuple[float, float]:
    """``(rtol, atol)`` for comparing tensors of ``dtype`` (a ``torch.dtype`` or its name)."""
    name = str(dtype).replace("torch.", "")
    return _resolve(name, rtol, atol, device, DTYPE_DEFAULTS.get(name, FALLBACK))


def for_check(check: str, device: Any = None, rtol=None, atol=None) -> tuple[float, float]:
    """``(rtol, atol)`` for one of ``CHECK_DEFAULTS``; explicit ``rtol``/``atol`` win."""
    return _resolve(check, rtol, atol, device, CHECK_DEFAULTS[check])
