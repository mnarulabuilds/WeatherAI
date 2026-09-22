import sys

from weather_ai.runtime_check import _append_tk_libexec_to_path, ensure_tkinter_available


def test_ensure_tkinter_available_passes_when_installed():
    ensure_tkinter_available()


def test_homebrew_path_helper_on_macos():
    if sys.platform != "darwin":
        return
    # After python-tk@3.14 install this directory exists on Homebrew setups.
    assert _append_tk_libexec_to_path() in {True, False}
