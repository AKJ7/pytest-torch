"""A ready-made test suite for a model: subclass, describe the model, get the tests.

    from pytest_torch import ModelSuite

    class TestClassifier(ModelSuite):
        checks = ("contract", "compile", "export", "training")

        def make_model(self):
            return Classifier()

        def make_inputs(self, make_tensor):
            return make_tensor(8, 16)

        def loss(self, output, inputs):  # only needed for "training"
            return output.pow(2).mean()

Every generated test runs on each device of ``torch_device``, seeded by ``torch_seed``.
"""
from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest


def _set_dotted(config: Any, dotted: str, value: Any) -> None:
    """``config["a"]["b"] = value`` for ``"a.b"``, on dicts or attribute objects."""
    *parents, last = dotted.split(".")
    node = config
    for key in parents:
        try:
            node = node[key] if isinstance(node, Mapping) else getattr(node, key)
        except (KeyError, AttributeError):
            raise KeyError(f"config_overrides: {dotted!r}: the config has no {key!r} there") from None
    if isinstance(node, dict):
        node[last] = value
    else:
        setattr(node, last, value)

# check name -> generated test methods
CHECKS: dict[str, tuple[str, ...]] = {
    "contract": ("test_inference_contract",),
    "compile": ("test_compile_parity",),
    "export": ("test_export_parity",),
    "torchscript": ("test_torchscript_parity",),
    "onnx": ("test_onnx_parity",),
    "aoti": ("test_aoti_parity",),
    "autocast": ("test_autocast_parity",),
    "golden": ("test_golden",),
    "output_spec": ("test_output_spec",),
    "checkpoint": ("test_checkpoint_roundtrip",),
    "layouts": ("test_layout_invariance",),
    "frozen": ("test_frozen",),
    "training": (
        "test_gradients_finite",
        "test_all_params_receive_grad",
        "test_loss_decreases",
        "test_overfits_one_batch",
    ),
}
DEFAULT_CHECKS = ("contract", "compile", "export")

# Defaults for `check_options`, merged under the user's values.
_DEFAULT_OPTIONS: dict[str, dict[str, Any]] = {
    "compile_parity": {"backend": "eager"},  # no C++ toolchain needed
}


class ModelSuite:
    """Base class for a generated model test suite (it is not collected itself).

    Override :meth:`make_model` and :meth:`make_inputs` (and :meth:`loss` for the
    ``"training"`` checks), and pick the checks with ``checks``; see ``CHECKS`` for all
    names. ``check_options`` passes keyword arguments to the assertion behind a test,
    keyed by the test name without ``test_``, e.g.
    ``{"inference_contract": {"skip": ["device_parity"]}, "overfits_one_batch": {"steps": 500}}``.
    Any generated test can be overridden like a normal method.
    """

    checks: ClassVar[tuple[str, ...]] = DEFAULT_CHECKS
    check_options: ClassVar[dict[str, dict[str, Any]]] = {}
    #: also export with a dynamic batch dim and run a smaller batch through the program
    dynamic_batch: ClassVar[bool] = False
    #: for the "output_spec" check: a spec as for assert_output_spec; "B" is the batch size
    output_spec: ClassVar[Any] = None
    #: for the "frozen" check: fnmatch patterns of parameters that must not train (see
    #: frozen_parameters() to make them depend on the config)
    frozen: ClassVar[tuple[str, ...]] = ()
    #: run every generated test once per config: paths (.yaml/.yml/.json) or objects; the
    #: current one is ``self.config`` (and the ``model_config`` fixture)
    configs: ClassVar[Sequence[Any]] = ()
    #: dotted keys set on every loaded config, e.g. to make it small enough to test:
    #: ``{"model.layers": 1, "env.num_environments": 1}``
    config_overrides: ClassVar[Mapping[str, Any]] = {}
    #: the current config while a generated test runs (None without ``configs``)
    config: Any = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        unknown = set(cls.checks) - set(CHECKS)
        if unknown:
            raise TypeError(f"{cls.__name__}: unknown check(s) {sorted(unknown)}; choose from {list(CHECKS)}")
        known_options = {m[len("test_"):] for methods in CHECKS.values() for m in methods}
        bad_options = set(cls.check_options) - known_options
        if bad_options:
            raise TypeError(
                f"{cls.__name__}: unknown check_options key(s) {sorted(bad_options)}; "
                f"keys are test names without 'test_': {sorted(known_options)}"
            )
        if "training" in cls.checks and cls.loss is ModelSuite.loss:
            raise TypeError(f"{cls.__name__}: the 'training' checks need a loss(self, output, inputs) method")
        if "configs" in cls.__dict__ and not list(cls.configs):
            raise TypeError(f"{cls.__name__}: configs is empty (check the path or glob that builds it)")
        if "frozen" in cls.checks and cls.loss is ModelSuite.loss:
            raise TypeError(f"{cls.__name__}: the 'frozen' check needs a loss(self, output, inputs) method")
        if "output_spec" in cls.checks and cls.output_spec is None:
            raise TypeError(f"{cls.__name__}: the 'output_spec' check needs an output_spec attribute")
        # Disabled checks disappear (None is not collected); enabled ones come back from
        # the base class unless this class defines its own version.
        for check, methods in CHECKS.items():
            for name in methods:
                if name in cls.__dict__:
                    continue
                setattr(cls, name, getattr(ModelSuite, name) if check in cls.checks else None)

    # ------------------------------------------------------------------ #
    # what users override
    # ------------------------------------------------------------------ #
    def make_model(self):
        """Return a fresh model. Weights are seeded by ``torch_seed``."""
        raise NotImplementedError(f"{type(self).__name__} must implement make_model()")

    def make_inputs(self, make_tensor) -> Any:
        """Return example inputs (tensor, tuple of args or dict of kwargs) on the test's device."""
        raise NotImplementedError(f"{type(self).__name__} must implement make_inputs(make_tensor)")

    def loss(self, output, inputs):
        """Return a scalar training loss for ``output``; needed by the ``"training"`` checks."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def frozen_parameters(self) -> Sequence[str]:
        """Patterns for the "frozen" check; override to depend on ``self.config`` (an empty
        result skips the check for that config)."""
        return self.frozen

    def load_config(self, source: Any) -> Any:
        """Turn an entry of ``configs`` into a config: YAML/JSON files are parsed (YAML needs
        PyYAML), anything else is returned as is. Override for Hydra/OmegaConf composition."""
        if isinstance(source, (str, os.PathLike)) and Path(source).suffix in (".yaml", ".yml", ".json"):
            path = Path(source)
            if path.suffix == ".json":
                return json.loads(path.read_text(encoding="utf-8"))
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("YAML configs need PyYAML: pip install pyyaml") from exc
            return yaml.safe_load(path.read_text(encoding="utf-8"))
        return source

    @classmethod
    def config_id(cls, source: Any) -> str:
        """Test id for a config: the file name without extension, or ``config<i>``."""
        if isinstance(source, (str, os.PathLike)):
            return Path(source).stem
        return f"config{list(cls.configs).index(source)}"

    def resolve_config(self, source: Any) -> Any:
        config = copy.deepcopy(self.load_config(source))
        for dotted, value in self.config_overrides.items():
            _set_dotted(config, dotted, value)
        return config

    def options(self, test: str) -> dict[str, Any]:
        return {**_DEFAULT_OPTIONS.get(test, {}), **self.check_options.get(test, {})}

    def _setup(self, torch_device, make_tensor, model_config=None, *, train: bool = False):
        from .repro import record_repro

        self.config = model_config
        model = self.make_model().to(torch_device)
        model = model.train() if train else model.eval()
        # Part of the failure bundle under --torch-repro-dir (weights if small enough).
        model = record_repro("model", model)
        return model, record_repro("inputs", self.make_inputs(make_tensor))

    def _loss_fn(self, inputs):
        return lambda output: self.loss(output, inputs)

    @staticmethod
    def _require_compile():
        from .utils import is_compile_supported

        if not is_compile_supported():
            pytest.skip("torch.compile / torch.export (dynamo) is not supported here")

    # ------------------------------------------------------------------ #
    # generated tests
    # ------------------------------------------------------------------ #
    def test_inference_contract(self, torch_seed, torch_device, make_tensor, model_config):
        from .contract import check_inference_contract

        model, inputs = self._setup(torch_device, make_tensor, model_config)
        check_inference_contract(model, inputs, **self.options("inference_contract"))

    def test_compile_parity(self, torch_seed, torch_device, make_tensor, model_config):
        from .parity import assert_compile_parity

        self._require_compile()
        model, inputs = self._setup(torch_device, make_tensor, model_config)
        assert_compile_parity(model, inputs, **self.options("compile_parity"))

    def test_export_parity(self, torch_seed, torch_device, make_tensor, model_config):
        from .parity import assert_export_parity
        from .utils import find_batch_size, tree_map

        self._require_compile()
        model, inputs = self._setup(torch_device, make_tensor, model_config)
        options = self.options("export_parity")
        more = []
        if self.dynamic_batch:
            batch = find_batch_size(inputs)
            smaller = (batch or 0) // 2
            if smaller < 2:
                pytest.fail("dynamic_batch=True needs example inputs with a batch size >= 4", pytrace=False)
            more.append(tree_map(lambda t: t[:smaller] if t.dim() >= 1 and t.shape[0] == batch else t, inputs))
            options.setdefault("dynamic_batch", True)
        assert_export_parity(model, inputs, *more, **options)

    def test_torchscript_parity(self, torch_seed, torch_device, make_tensor, model_config):
        from .parity import assert_torchscript_parity

        model, inputs = self._setup(torch_device, make_tensor, model_config)
        assert_torchscript_parity(model, inputs, **self.options("torchscript_parity"))

    def test_onnx_parity(self, torch_seed, torch_device, make_tensor, model_config):
        from .parity import assert_onnx_parity

        pytest.importorskip("onnxruntime")
        model, inputs = self._setup(torch_device, make_tensor, model_config)
        assert_onnx_parity(model, inputs, **self.options("onnx_parity"))

    def test_aoti_parity(self, torch_seed, torch_device, make_tensor, model_config):
        from .parity import assert_aoti_parity

        self._require_compile()
        model, inputs = self._setup(torch_device, make_tensor, model_config)
        assert_aoti_parity(model, inputs, **self.options("aoti_parity"))

    def test_autocast_parity(self, torch_seed, torch_device, make_tensor, model_config):
        from .parity import assert_autocast_parity

        model, inputs = self._setup(torch_device, make_tensor, model_config)
        assert_autocast_parity(model, inputs, **self.options("autocast_parity"))

    def test_golden(self, request, torch_seed, torch_device, make_tensor, model_config):
        from .golden import Golden
        from .utils import run_inference

        model, inputs = self._setup(torch_device, make_tensor, model_config)
        name = type(self).__name__
        callspec = getattr(request.node, "callspec", None)
        if callspec is not None and "model_config" in callspec.params:  # one golden file per config
            name += f"[{self.config_id(callspec.params['model_config'])}]"
        path = request.node.path.parent / "__golden__" / f"{name}.pt"
        golden = Golden(path, update=request.config.getoption("torch_update_golden"))
        golden.check(run_inference(model, inputs), **{"atol": 1e-4, **self.options("golden")})

    def test_output_spec(self, torch_seed, torch_device, make_tensor, model_config):
        from .specs import assert_output_spec
        from .utils import find_batch_size, run_inference

        model, inputs = self._setup(torch_device, make_tensor, model_config)
        dims = {"B": find_batch_size(inputs)} if find_batch_size(inputs) is not None else {}
        assert_output_spec(run_inference(model, inputs), self.output_spec, **{**dims, **self.options("output_spec")})

    def test_checkpoint_roundtrip(self, torch_seed, torch_device, make_tensor, model_config):
        from .checkpoint import assert_checkpoint_roundtrip

        model, inputs = self._setup(torch_device, make_tensor, model_config)
        assert_checkpoint_roundtrip(
            lambda: self.make_model().to(torch_device), inputs, model=model, **self.options("checkpoint_roundtrip")
        )

    def test_layout_invariance(self, torch_seed, torch_device, make_tensor, model_config):
        from .invariance import assert_layout_invariant

        model, inputs = self._setup(torch_device, make_tensor, model_config)
        assert_layout_invariant(model, inputs, **self.options("layout_invariance"))

    def test_frozen(self, torch_seed, torch_device, make_tensor, model_config):
        from .training import assert_frozen

        model, inputs = self._setup(torch_device, make_tensor, model_config)
        patterns = list(self.frozen_parameters())
        if not patterns:
            pytest.skip("nothing is frozen in this configuration")
        assert_frozen(model, inputs, self._loss_fn(inputs), patterns, **self.options("frozen"))

    def test_gradients_finite(self, torch_seed, torch_device, make_tensor, model_config):
        from .training import assert_gradients_finite

        model, inputs = self._setup(torch_device, make_tensor, model_config, train=True)
        assert_gradients_finite(model, inputs, self._loss_fn(inputs), **self.options("gradients_finite"))

    def test_all_params_receive_grad(self, torch_seed, torch_device, make_tensor, model_config):
        from .training import assert_all_params_receive_grad

        model, inputs = self._setup(torch_device, make_tensor, model_config, train=True)
        assert_all_params_receive_grad(model, inputs, self._loss_fn(inputs), **self.options("all_params_receive_grad"))

    def test_loss_decreases(self, torch_seed, torch_device, make_tensor, model_config):
        from .training import assert_loss_decreases

        model, inputs = self._setup(torch_device, make_tensor, model_config)
        assert_loss_decreases(model, inputs, self._loss_fn(inputs), **self.options("loss_decreases"))

    def test_overfits_one_batch(self, torch_seed, torch_device, make_tensor, model_config):
        from .training import assert_overfits_one_batch

        model, inputs = self._setup(torch_device, make_tensor, model_config)
        assert_overfits_one_batch(model, inputs, self._loss_fn(inputs), **self.options("overfits_one_batch"))
