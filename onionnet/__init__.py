__all__ = ["OnionNet", "__version__"]

# Expose package version that matches installed metadata
try:  # Python 3.8+
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version
except Exception:  # pragma: no cover - fallback if needed
    try:
        from importlib_metadata import PackageNotFoundError  # type: ignore
        from importlib_metadata import version as _pkg_version
    except Exception:  # Last resort
        PackageNotFoundError = Exception  # type: ignore
        _pkg_version = lambda name: "0.0.0"  # type: ignore

try:
    __version__ = _pkg_version("onionnet")
except PackageNotFoundError:  # when running from source without install
    __version__ = "0.0.0"


def __getattr__(name):
    # Lazy import to avoid requiring heavy optional deps (graph_tool) on import
    if name == "OnionNet":
        from .onionnet import OnionNet as _OnionNet

        return _OnionNet
    raise AttributeError(f"module 'onionnet' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + ["OnionNet"])
