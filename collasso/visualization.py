"""
collasso.visualization — animazione e grafici riassuntivi della simulazione
del collasso gravitazionale (Step 7).

`matplotlib.use("Agg")` va impostato PRIMA di qualunque altro import di
matplotlib (obbligatorio per evitare errori su sistemi headless senza
display) — per questo e' la primissima riga eseguibile del modulo.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from collasso.pipeline import SimulationResult  # noqa: E402

# Numero massimo di frame dell'animazione: se CollapseSolution.t_s ne
# contiene di piu', si sottocampiona con np.linspace di INDICI (mai uno
# slicing arbitrario che rischi di saltare l'istante finale).
N_FRAMES_MAX = 100


def _frame_indices(n_tempi: int, n_frames_max: int = N_FRAMES_MAX) -> np.ndarray:
    """Indici di sottocampionamento uniforme in indice (non in tempo), che
    includono sempre il primo e l'ultimo istante disponibile.
    """
    if n_tempi <= n_frames_max:
        return np.arange(n_tempi)
    return np.unique(np.linspace(0, n_tempi - 1, n_frames_max).astype(int))


def animate_collapse(result: SimulationResult, output_path: str | Path) -> Path:
    """Anima l'evoluzione del collasso: raggio (km) di ogni shell vs
    frazione di massa racchiusa (m_enclosed_g/M_tot), un frame per istante
    campionato. Titolo con tempo corrente (ms) e densita' centrale corrente
    (dalla shell piu' interna, `rho_c_gcm3_t`). Salvata come GIF con
    `PillowWriter` (nessuna dipendenza da ffmpeg).

    Crea le directory intermedie di `output_path` se non esistono. Ritorna
    il path assoluto scritto.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sol = result.collapse_solution
    m_tot_g = float(result.m_enclosed_g[-1])
    frazione_massa = result.m_enclosed_g / m_tot_g

    indici = _frame_indices(len(sol.t_s))

    fig, ax = plt.subplots(figsize=(6, 5))
    (linea,) = ax.plot([], [], "o-", markersize=2, linewidth=1)
    ax.set_xlabel("Frazione di massa racchiusa (m_enclosed / M_tot)")
    ax.set_ylabel("Raggio della shell (km)")

    r_max_km = float(np.max(sol.r_cm[:, 0]) / 1.0e5) * 1.1
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, max(r_max_km, 1.0))

    def _update(frame_idx: int):
        i = indici[frame_idx]
        r_km = sol.r_cm[:, i] / 1.0e5
        linea.set_data(frazione_massa, r_km)
        t_ms = sol.t_s[i] * 1.0e3
        rho_c = sol.rho_c_gcm3_t[i]
        ax.set_title(
            f"Collasso di {result.star.id}: t={t_ms:.4f} ms, "
            f"rho_c={rho_c:.3e} g/cm^3"
        )
        return (linea,)

    anim = FuncAnimation(fig, _update, frames=len(indici), blit=False)
    anim.save(str(output_path), writer=PillowWriter(fps=10))
    plt.close(fig)

    return output_path


def plot_summary(result: SimulationResult, output_path: str | Path) -> Path:
    """Figura statica a 3 pannelli:
        (a) densita' centrale (shell piu' interna) vs tempo (scala log su y);
        (b) velocita' della shell piu' interna vs tempo;
        (c) curva massa-raggio TOV per SLy e APR4 (riusando le sequenze
            gia' calcolate in `SimulationResult`, nessun ricalcolo), con un
            marker orizzontale sulla massa del nucleo della stella scelta.

    Crea le directory intermedie di `output_path` se non esistono. Ritorna
    il path assoluto scritto.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sol = result.collapse_solution
    t_ms = sol.t_s * 1.0e3

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) densita' centrale vs tempo (log y)
    ax_rho = axes[0]
    ax_rho.plot(t_ms, sol.rho_c_gcm3_t)
    ax_rho.set_yscale("log")
    ax_rho.set_xlabel("t (ms)")
    ax_rho.set_ylabel("rho_c (g/cm^3, shell piu' interna)")
    ax_rho.set_title("Densita' centrale vs tempo")

    # (b) velocita' della shell piu' interna vs tempo
    ax_v = axes[1]
    ax_v.plot(t_ms, sol.v_cms[0, :] / 1.0e5)
    ax_v.set_xlabel("t (ms)")
    ax_v.set_ylabel("v shell piu' interna (km/s)")
    ax_v.set_title("Velocita' shell piu' interna vs tempo")

    # (c) curva massa-raggio TOV (SLy, APR4) + marker massa stella scelta
    ax_mr = axes[2]
    for nome, sequenza in (("SLy", result.tov_sequence_sly), ("APR4", result.tov_sequence_apr4)):
        r_km = np.array([s.R_cm / 1.0e5 for s in sequenza])
        m_msun = np.array([s.M_msun for s in sequenza])
        ax_mr.plot(r_km, m_msun, marker=".", label=nome)
    ax_mr.axhline(
        result.star.massa_nucleo_msun,
        color="black",
        linestyle="--",
        label=f"massa nucleo {result.star.id}",
    )
    ax_mr.set_xlabel("R (km)")
    ax_mr.set_ylabel("M (Msun)")
    ax_mr.set_title("Curva massa-raggio TOV (SLy/APR4)")
    ax_mr.legend(fontsize="small")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path
