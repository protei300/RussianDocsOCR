"""Optional per-stage instrumentation for cross-language conformance testing.

The library is being reimplemented in Go, .NET, Kotlin and C++. Comparing only
the final result of a port against this reference tells you *that* it diverged,
never *where* — and a whole-pipeline mismatch on a twelve-model pipeline is a
week of bisection. Comparing intermediate stages turns that into "first
divergence at ``fields.bbox``", which is an hour.

This module is the mechanism. ``Pipeline.probe`` is ``None`` in production, so
every emission site costs one attribute test and nothing else: no payload is
built, no copy is made, no branch is taken beyond the guard. Nothing here changes
what ``process_img`` computes or returns.

Two design constraints are deliberate and should survive refactoring:

* **Do not thread dump values through return types.** Making stage functions
  return extra data, or adding parameters they pass along, would alter the very
  code the ports are transliterating — and then the ports would be copying the
  instrumentation rather than the algorithm.
* **Emit references, not deep copies.** A sink that needs to keep a payload past
  the call is responsible for copying it. The pipeline reuses and rebinds arrays
  (``self.results`` is rebound on every ``process_img``), so a sink that stores
  a reference and reads it later gets whatever that array became — the same trap
  the service hit with ``PipelineResults``.

Stage names are a fixed, ordered vocabulary defined in
``conformance/spec/stages.md``. They may be added to but never renamed: the
golden files are keyed by them.

Usage::

    from document_processing.pipeline.probe import DirectoryStageSink

    with DirectoryStageSink('dump/') as sink:
        pipeline.probe = sink
        try:
            results = pipeline.process_img(path)
        finally:
            pipeline.probe = None
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class StageSink(Protocol):
    """Receives one named intermediate value per pipeline stage.

    A single method on purpose. Anything richer (filtering, ordering, buffering)
    belongs in the implementation, not in the interface the pipeline depends on.
    """

    def emit(self, name: str, payload: Any) -> None:  # pragma: no cover - protocol
        ...


class NullStageSink:
    """A sink that discards everything.

    Exists so tests can exercise the emission sites without writing files.
    Production uses ``probe = None`` instead, which skips the calls entirely.
    """

    def emit(self, name: str, payload: Any) -> None:
        pass


class RecordingStageSink:
    """Keeps every payload in memory, in emission order.

    Payloads are stored **as passed**, without copying, matching the contract in
    this module's docstring. For arrays the pipeline may later rebind or mutate,
    pass ``copy=True``.
    """

    def __init__(self, copy: bool = False):
        self.copy = copy
        self.stages: list[tuple[str, Any]] = []

    def emit(self, name: str, payload: Any) -> None:
        if self.copy and isinstance(payload, np.ndarray):
            payload = payload.copy()
        self.stages.append((name, payload))

    def names(self) -> list[str]:
        return [n for n, _ in self.stages]

    def get(self, name: str) -> Any:
        for n, p in self.stages:
            if n == name:
                return p
        raise KeyError(name)


class DirectoryStageSink:
    """Writes each stage to ``<root>/<name>.npy`` or ``<name>.json``.

    ndarrays become ``.npy`` (self-describing: dtype, shape and order travel with
    the bytes, so a transposed payload fails loudly instead of comparing equal).
    Everything else becomes JSON, with numpy scalars and arrays coerced, because
    ``json.dumps`` cannot serialise them and a port would otherwise have nothing
    to compare against.

    An ordered index is written to ``stages.json`` so a consumer knows which
    stages ran — which is how ``--upto`` makes a partial port verifiable.
    """

    def __init__(self, root: str | Path, upto: str | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.upto = upto
        self.index: list[dict[str, Any]] = []
        self._stopped = False

    def emit(self, name: str, payload: Any) -> None:
        if self._stopped:
            return

        safe = name.replace("/", "_")
        if isinstance(payload, np.ndarray):
            path = self.root / f"{safe}.npy"
            np.save(path, payload, allow_pickle=False)
            entry = {"stage": name, "file": path.name, "kind": "npy",
                     "dtype": str(payload.dtype), "shape": list(payload.shape)}
        else:
            path = self.root / f"{safe}.json"
            path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False),
                            encoding="utf-8")
            entry = {"stage": name, "file": path.name, "kind": "json"}
        self.index.append(entry)

        # `--upto` stops after the named stage rather than before, so the stage
        # the caller asked about is included.
        if self.upto is not None and name == self.upto:
            self._stopped = True

    def close(self) -> None:
        (self.root / "stages.json").write_text(
            json.dumps({"upto": self.upto, "stopped_early": self._stopped,
                        "stages": self.index}, indent=2, ensure_ascii=False),
            encoding="utf-8")

    def __enter__(self) -> "DirectoryStageSink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _jsonable(value: Any) -> Any:
    """Coerce numpy types so ``json.dumps`` succeeds without a ``default=`` hook.

    Deliberately explicit rather than relying on a fallback hook: an unexpected
    type should be visible here, not silently stringified into a golden file that
    a port can never reproduce.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
