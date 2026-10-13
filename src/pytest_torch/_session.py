"""Settings of the running pytest session that library code may consult.

Torch-free: the plugin writes it at startup; library functions read it, and work the
same outside pytest (where nothing is set).
"""
from __future__ import annotations

#: devices pinned with --torch-device (``None``: not pinned, use every available device)
selected_devices: list[str] | None = None
