"""CAVIS: controlled audits of validity and invariance scores."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cavis")
except PackageNotFoundError:  # pragma: no cover - editable source tree
    __version__ = "0.1.0"

__all__ = ["__version__"]
