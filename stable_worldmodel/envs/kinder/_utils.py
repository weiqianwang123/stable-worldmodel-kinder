"""Shared helpers for KinDER environment adapters."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


DEFAULT_KINDERGARDEN_HOME = Path('/home/robin_wang/kindergarden')
KINDERGARDEN_HOME_ENV = 'KINDERGARDEN_HOME'


def ensure_kindergarden_on_path(
    home: str | os.PathLike | None = None,
) -> None:
    """Make a local KinDER checkout importable if it is present."""
    root = Path(
        home
        or os.getenv(KINDERGARDEN_HOME_ENV)
        or DEFAULT_KINDERGARDEN_HOME
    ).expanduser()
    src = root / 'src'
    if src.is_dir():
        src_str = str(src)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)

        kinder_pkg = sys.modules.get('kinder')
        pkg_path = getattr(kinder_pkg, '__path__', None)
        if pkg_path is not None:
            kinder_src = str(src / 'kinder')
            if kinder_src not in pkg_path:
                pkg_path.insert(0, kinder_src)


def load_kindergarden_class(
    module_path: str,
    class_name: str,
    *,
    env_label: str,
    dependency_hint: str,
    home: str | os.PathLike | None = None,
):
    """Load a KinDER class, falling back to a local kindergarden checkout."""
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except (ImportError, AttributeError):
        ensure_kindergarden_on_path(home)
        try:
            mod = importlib.import_module(module_path)
            return getattr(mod, class_name)
        except (ImportError, AttributeError) as local_exc:
            raise ImportError(
                f'Could not import {env_label}. Install '
                f'`{dependency_hint}`, or set KINDERGARDEN_HOME to a local '
                'kindergarden checkout with the required dependencies.'
            ) from local_exc


__all__ = [
    'DEFAULT_KINDERGARDEN_HOME',
    'KINDERGARDEN_HOME_ENV',
    'ensure_kindergarden_on_path',
    'load_kindergarden_class',
]
