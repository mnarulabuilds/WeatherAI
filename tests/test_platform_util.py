import sys
from unittest.mock import MagicMock, patch

import pytest

from weather_ai.platform_util import open_path_in_file_manager, parse_year_range


def test_parse_year_range_defaults():
    assert parse_year_range("", "", 1997, 2015) == (1997, 2015)


def test_parse_year_range_invalid():
    with pytest.raises(ValueError):
        parse_year_range("2015", "2010", 1997, 2015)


@patch("weather_ai.platform_util.subprocess.run")
def test_open_path_macos(mock_run, tmp_path, monkeypatch):
    file_path = tmp_path / "plot.png"
    file_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sys, "platform", "darwin")
    open_path_in_file_manager(str(file_path))
    mock_run.assert_called_once()


@patch("weather_ai.platform_util.subprocess.run")
def test_open_path_linux(mock_run, tmp_path, monkeypatch):
    file_path = tmp_path / "plot.png"
    file_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sys, "platform", "linux")
    open_path_in_file_manager(str(file_path))
    mock_run.assert_called_once()


@patch("weather_ai.platform_util.os.startfile", create=True)
def test_open_path_windows(mock_startfile, tmp_path, monkeypatch):
    file_path = tmp_path / "plot.png"
    file_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sys, "platform", "win32")
    open_path_in_file_manager(str(file_path))
    mock_startfile.assert_called_once()
