"""Settings of the running pytest session that library code may consult.

Torch-free: the plugin writes it at startup; library functions read it, and work the
same outside pytest (where nothing is set).
"""
from __future__ import annotations

#: devices pinned with --torch-device (``None``: not pinned, use every available device)
selected_devices: list[str] | None = None

# zero-code testing (--torch-model)
model_specs: list[str] = []
model_checks: tuple[str, ...] = ()
model_inputs: str | None = None
model_pretrained: bool = False
#: --torch-model without file arguments: run only the zero-code tests
zero_code_only: bool = False
