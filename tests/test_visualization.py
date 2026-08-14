"""
tests/test_visualization.py — test di `collasso.visualization` (Step 7).

Calcola UNA `SimulationResult` (s20) a livello di modulo (riusata fra i
test per non rallentare la suite), poi verifica che `animate_collapse`/
`plot_summary` producano file con dimensione > 0 byte.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collasso.pipeline import run_full_simulation
from collasso.visualization import animate_collapse, plot_summary

# Fixture-style a livello di modulo: una sola SimulationResult riusata da
# entrambi i test, per non rallentare la suite (pipeline completa per s20).
_RESULT = run_full_simulation("s20")


def test_animate_collapse_creates_nonempty_gif():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "sottocartella" / "collapse_s20.gif"
        percorso_scritto = animate_collapse(_RESULT, output_path)

        assert percorso_scritto.exists()
        assert os.path.getsize(percorso_scritto) > 0


def test_plot_summary_creates_nonempty_png():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "sottocartella" / "summary_s20.png"
        percorso_scritto = plot_summary(_RESULT, output_path)

        assert percorso_scritto.exists()
        assert os.path.getsize(percorso_scritto) > 0
