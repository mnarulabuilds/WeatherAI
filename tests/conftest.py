from __future__ import annotations

import pytest


def write_weather_year(path, year: int, rows: int = 365) -> None:
    codes = ["0001", "0010", "0100", "1000"]
    lines = []
    for day in range(rows):
        base = 10 + (day % 15)
        code = codes[day % len(codes)]
        values = [1, base + 5, base, base + 2, base - 1, 90, 50, 1015, 1010, 5, 2, 10 + (day % 5), code]
        lines.append(" ".join(str(v) for v in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def sample_data_dir(tmp_path):
    for year in (2001, 2002, 2003):
        write_weather_year(tmp_path / f"Weather{year}.txt", year)
    return tmp_path


@pytest.fixture
def fast_config():
    from weather_ai.config import ModelConfig

    return ModelConfig(
        reg_max_iter=80,
        clf_max_iter=80,
        reg_hidden_layers=(8,),
        clf_hidden_layers=(8,),
        holdout_fraction=0.2,
    )
