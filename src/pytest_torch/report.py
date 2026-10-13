"""Better failure output: mismatch heatmaps and GitHub Actions annotations.

Torch-free at import time (torch/matplotlib are imported lazily when a heatmap is drawn).
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

_state: dict[str, Any] = {"dir": None, "test": None, "warned": False}


def configure(artifact_dir: str | None) -> None:
    _state["dir"] = artifact_dir


def set_current_test(nodeid: str | None) -> None:
    _state["test"] = nodeid


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:100] or "x"


def _to_2d(t):
    if t.dim() == 0:
        return t.reshape(1, 1)
    if t.dim() == 1:
        return t.reshape(1, -1)
    return t.reshape(-1, t.shape[-1])


def _downsample(grid, limit: int):
    import torch

    rows, cols = grid.shape
    if rows <= limit and cols <= limit:
        return grid
    pooled = torch.nn.functional.adaptive_max_pool2d(
        grid[None, None], (min(rows, limit), min(cols, limit))
    )
    return pooled[0, 0]


def save_mismatch_heatmap(
    actual, expected, path, *, rtol: float, atol: float, title: str = ""
) -> Path | None:
    """Save a PNG showing where ``actual`` deviates from ``expected``.

    Colors show ``log10(|diff| / (atol + rtol*|expected|))``: red (> 0) is outside the
    tolerance. Large tensors are max-pooled to at most 256x256 so no mismatch disappears.
    Returns ``None`` (with a one-time warning) if matplotlib is not installed.
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError:
        if not _state["warned"]:
            _state["warned"] = True
            warnings.warn(
                "pytest-torch: install matplotlib (pip install 'pytest-torch[plots]') "
                "to get mismatch heatmaps",
                stacklevel=2,
            )
        return None
    import torch

    a = actual.detach().to("cpu", torch.float64)
    e = expected.detach().to("cpu", torch.float64)
    ratio = (a - e).abs() / (atol + rtol * e.abs()).clamp_min(1e-300)
    ratio = torch.nan_to_num(ratio, nan=1e3, posinf=1e3)
    same = (a == e) | (torch.isnan(a) & torch.isnan(e))
    ratio = torch.where(same, torch.zeros_like(ratio), ratio)
    n_bad = int((ratio > 1.0).sum())
    grid = _downsample(_to_2d(torch.log10(ratio.clamp(1e-3, 1e3))), 256)

    figure = Figure(figsize=(7.5, 5.0))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    image = axes.imshow(
        grid.numpy(), aspect="auto", cmap="coolwarm", vmin=-3.0, vmax=3.0, interpolation="nearest"
    )
    heading = f"{n_bad}/{a.numel()} elements beyond tolerance (shape {tuple(a.shape)})"
    axes.set_title(f"{title}\n{heading}" if title else heading, fontsize=9)
    figure.colorbar(image, ax=axes, label="log10(|diff| / (atol + rtol*|expected|)); > 0 = mismatch")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    return path


def record_mismatch(path: str, actual, expected, rtol: float, atol: float) -> str | None:
    """Save a heatmap for a failed comparison if ``--torch-failure-artifacts`` is set."""
    directory = _state["dir"]
    if directory is None:
        return None
    stem = f"{_slug(_state['test'] or 'outside-test')}__{_slug(path)}"
    target = Path(directory) / f"{stem}.png"
    counter = 2
    while target.exists():
        target = Path(directory) / f"{stem}_{counter}.png"
        counter += 1
    saved = save_mismatch_heatmap(
        actual, expected, target, rtol=rtol, atol=atol, title=f"{_state['test'] or ''} {path}".strip()
    )
    return None if saved is None else str(saved)


def _escape_data(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    return _escape_data(text).replace(":", "%3A").replace(",", "%2C")


def github_annotation(
    *, title: str, message: str, file: str | None = None, line: int | None = None,
    level: str = "error",
) -> str:
    """Format a GitHub Actions workflow command, e.g. ``::error file=a.py,line=3,title=t::msg``."""
    props = []
    if file:
        props.append(f"file={_escape_property(file)}")
    if line:
        props.append(f"line={line}")
    props.append(f"title={_escape_property(title)}")
    return f"::{level} {','.join(props)}::{_escape_data(message)}"
