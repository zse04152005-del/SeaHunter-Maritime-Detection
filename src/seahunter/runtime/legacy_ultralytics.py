"""Strict loader for the temporary repository-local Ultralytics baseline."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def legacy_source_path(repo_root: Path | None = None) -> Path:
    """Return and validate the local source directory used by the legacy detector."""

    root = (repo_root or _default_repo_root()).resolve()
    source = root / "yolo_source"
    package_init = source / "ultralytics" / "__init__.py"
    if not package_init.is_file():
        raise FileNotFoundError(f"legacy Ultralytics package not found: {package_init}")
    return source


def activate_legacy_ultralytics(repo_root: Path | None = None) -> ModuleType:
    """Import Ultralytics and fail if Python resolves a non-local installation."""

    source = legacy_source_path(repo_root)
    runtime_root = source.parent / ".runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime_root))
    already_loaded = sys.modules.get("ultralytics")
    if already_loaded is not None:
        loaded_name = getattr(already_loaded, "__file__", None)
        if loaded_name is None:
            raise RuntimeError("the loaded Ultralytics module has no filesystem location")
        loaded_file = Path(loaded_name).resolve()
        if source not in loaded_file.parents:
            raise RuntimeError(f"a non-local Ultralytics package is already loaded: {loaded_file}")
        return already_loaded

    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    module = importlib.import_module("ultralytics")
    loaded_name = module.__file__
    if loaded_name is None:
        raise RuntimeError("the imported Ultralytics module has no filesystem location")
    loaded_file = Path(loaded_name).resolve()
    if source not in loaded_file.parents:
        raise RuntimeError(f"expected local Ultralytics under {source}, loaded {loaded_file}")
    if getattr(module, "__version__", None) != "8.3.234":
        raise RuntimeError(f"unexpected legacy Ultralytics version: {getattr(module, '__version__', None)}")
    return module
