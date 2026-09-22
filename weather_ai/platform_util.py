import os
import subprocess
import sys


def open_path_in_file_manager(path: str) -> None:
    """Open a file or folder in the system file manager (cross-platform)."""
    absolute = os.path.abspath(path)
    if sys.platform == "win32":
        os.startfile(absolute)  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.run(["open", absolute], check=False)
        return
    subprocess.run(["xdg-open", absolute], check=False)


def parse_year_range(start_text: str, end_text: str, default_start: int, default_end: int) -> tuple[int, int]:
    start = int(start_text.strip()) if start_text.strip() else default_start
    end = int(end_text.strip()) if end_text.strip() else default_end
    if start > end:
        raise ValueError("Start year must be less than or equal to end year.")
    return start, end
