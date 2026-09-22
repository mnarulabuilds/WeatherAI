from __future__ import annotations

import os
import sys
from pathlib import Path


def _homebrew_tk_libexec_paths(version: str) -> list[Path]:
    prefix = os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew")
    return [
        Path(prefix) / f"opt/python-tk@{version}" / "libexec",
        Path("/usr/local") / f"opt/python-tk@{version}" / "libexec",
    ]


def _append_tk_libexec_to_path() -> bool:
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    for libexec in _homebrew_tk_libexec_paths(version):
        tk_module = libexec / f"_tkinter.cpython-{sys.version_info.major}{sys.version_info.minor}-darwin.so"
        if not tk_module.exists():
            # Linux/other naming
            candidates = list(libexec.glob("_tkinter*.so"))
            if not candidates:
                continue
        path_str = str(libexec)
        if path_str not in sys.path:
            sys.path.append(path_str)
        return True
    return False


def ensure_tkinter_available() -> None:
    """Ensure Tkinter can load; apply Homebrew path fix when python-tk is installed but not on sys.path."""
    try:
        import _tkinter  # noqa: F401
        return
    except ModuleNotFoundError:
        if sys.platform == "darwin" and _append_tk_libexec_to_path():
            try:
                import _tkinter  # noqa: F401
                return
            except ModuleNotFoundError:
                pass

    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    executable = sys.executable
    hints = [
        "WeatherAI's desktop UI requires Tkinter, but this Python build has no _tkinter module.",
        f"  Python: {executable} ({version})",
        "",
        "Fix on macOS (Homebrew):",
        f"  brew install python-tk@{version}",
        "  # If Tk was installed after creating the venv, recreate it:",
        "  rm -rf .venv && python3 -m venv .venv && source .venv/bin/activate",
        "  pip install -r requirements.txt",
        "",
        "Fix on Ubuntu/Debian:",
        "  sudo apt install python3-tk",
        "",
        "CLI alternative (no GUI):",
        "  python weather_engine.py",
    ]
    raise SystemExit("\n".join(hints))
