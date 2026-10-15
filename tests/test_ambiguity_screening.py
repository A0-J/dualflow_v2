import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "experiments" / "ambiguity_screening" / "screen_signals.py"
_spec = importlib.util.spec_from_file_location("screen_signals", _path)
screen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(screen)


def test_auroc():
    assert screen.auroc([3, 4], [1, 2]) == 1.0
    assert screen.auroc([1, 2], [3, 4]) == 0.0
    assert screen.auroc([1, 1], [1, 1]) == 0.5


def test_total_variation():
    assert screen.total_variation(["a", "a"], ["a", "a"]) == 0.0
    assert screen.total_variation(["a", "a"], ["b", "b"]) == 1.0
    assert screen.total_variation(["a", "b"], ["a", "a"]) == pytest.approx(0.5)
