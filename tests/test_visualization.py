"""
tests/test_visualization.py — test di `collasso.visualization` (Step 7;
esteso in un ciclo successivo puramente estetico con `animate_collapse_disk`
— vedi STATUS.md).

Calcola UNA `SimulationResult` (s20) a livello di modulo (riusata fra i
test per non rallentare la suite), poi verifica che `animate_collapse`/
`plot_summary`/`animate_collapse_disk` producano file con dimensione > 0
byte, che il testo dell'ultimo fotogramma di `animate_collapse_disk`
dichiari l'assenza di bounce, e che la densita' di shell derivata per il
colore sia fisicamente sensata (massa implicita coerente).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collasso.pipeline import run_full_simulation
from collasso.visualization import animate_collapse, animate_collapse_disk, plot_summary

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


def test_animate_collapse_disk_creates_nonempty_gif():
    """`animate_collapse_disk` (miglioramento estetico, nessuna nuova
    fisica) affianca `animate_collapse` senza sostituirla: file distinto,
    stessa struttura di test (GIF non vuota, path creato).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "sottocartella" / "collapse_disk_s20.gif"
        percorso_scritto = animate_collapse_disk(_RESULT, output_path)

        assert percorso_scritto.exists()
        assert os.path.getsize(percorso_scritto) > 0


def test_testo_fine_simulazione_dichiara_assenza_bounce():
    """Il testo sovrapposto sull'ultimo fotogramma di
    `animate_collapse_disk` (costante di modulo `TESTO_FINE_SIMULAZIONE`,
    riusata dalla funzione stessa, non duplicata qui) deve dichiarare
    esplicitamente sia che la simulazione termina li' sia l'assenza di un
    bounce modellato — richiesta esplicita dell'utente.
    """
    from collasso.visualization import TESTO_FINE_SIMULAZIONE

    testo_lower = TESTO_FINE_SIMULAZIONE.lower()
    assert "nessun bounce" in testo_lower
    assert "terminat" in testo_lower


def test_shell_densities_gcm3_massa_totale_coerente():
    """`_shell_densities_gcm3` (grandezza derivata SOLO per il colore
    dell'animazione 2D, nessuna nuova fisica) deve dare densita' positive
    e una massa totale implicita (densita' * volume, sommata su tutte le
    shell) coerente con `delta_m_g` — controllo di sanita' sulla formula,
    non sulla dinamica.
    """
    from collasso.visualization import _shell_densities_gcm3

    sol = _RESULT.collapse_solution
    delta_m_g = _RESULT.delta_m_g
    r_colonna = sol.r_cm[:, 0]

    densita = _shell_densities_gcm3(r_colonna, delta_m_g)
    assert np.all(densita > 0.0)

    r_full = np.concatenate(([0.0], r_colonna))
    volumi = (4.0 / 3.0) * np.pi * (r_full[1:] ** 3 - r_full[:-1] ** 3)
    massa_implicita_g = np.sum(densita * volumi)
    massa_attesa_g = np.sum(delta_m_g)
    assert massa_implicita_g == pytest.approx(massa_attesa_g, rel=1e-9)
