"""
collasso.visualization — animazione e grafici riassuntivi della simulazione
del collasso gravitazionale (Step 7; `animate_collapse_disk` aggiunta in un
ciclo successivo, puramente estetico — vedi STATUS.md).

`matplotlib.use("Agg")` va impostato PRIMA di qualunque altro import di
matplotlib (obbligatorio per evitare errori su sistemi headless senza
display) — per questo e' la primissima riga eseguibile del modulo.

Due animazioni distinte, entrambe generate da `scripts/run_simulation.py`,
nessuna sostituisce l'altra: `animate_collapse` (grafico a linea, raggio di
shell vs frazione di massa) e `animate_collapse_disk` (disco 2D pieno,
colorato per densita' di shell, pensato per dare l'impressione visiva di
una stella che collassa). Entrambe usano ESATTAMENTE gli stessi dati gia'
calcolati da `collasso.dynamics` (`CollapseSolution.r_cm`) — nessuna nuova
fisica, solo due modi diversi di visualizzare lo stesso risultato.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from collasso.pipeline import SimulationResult  # noqa: E402

# Numero massimo di frame dell'animazione: se CollapseSolution.t_s ne
# contiene di piu', si sottocampiona con np.linspace di INDICI (mai uno
# slicing arbitrario che rischi di saltare l'istante finale).
N_FRAMES_MAX = 100

# Testo sovrapposto sull'ULTIMO fotogramma di `animate_collapse_disk`
# (richiesta esplicita dell'utente): dichiara esplicitamente che la
# simulazione termina li' e che nessun bounce e' modellato — stesso
# limite di modello gia' dichiarato nei disclaimer (`collasso.pipeline.
# _build_disclaimers`, voce 4), qui reso visivamente esplicito. Costante
# di modulo (non una stringa inline) cosi' da essere testabile
# direttamente, senza dover fare OCR su un frame di GIF salvata.
TESTO_FINE_SIMULAZIONE = "Simulazione terminata qui - nessun bounce modellato"


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


def _shell_densities_gcm3(r_cm_colonna: np.ndarray, delta_m_g: np.ndarray) -> np.ndarray:
    """Densita' di ciascuna shell (g/cm^3) a un SINGOLO istante, dai raggi
    di bordo `r_cm_colonna` (shape (N,), una singola colonna temporale di
    `CollapseSolution.r_cm`) e dalle masse di shell fisse `delta_m_g`.

    Migliramento puramente estetico (nessuna nuova fisica): stessa formula
    gia' usata in `collasso.dynamics._shell_dvdt` per ricavare rho dal
    raggio di bordo (raggio -> volume di shell -> densita' media),
    duplicata qui perche' e' una grandezza derivata di post-processing
    usata SOLO per colorare `animate_collapse_disk`, mai riletta dalla
    dinamica ne' influente su di essa.
    """
    r_full = np.concatenate(([0.0], r_cm_colonna))
    volumi = (4.0 / 3.0) * np.pi * (r_full[1:] ** 3 - r_full[:-1] ** 3)
    return delta_m_g / np.maximum(volumi, 1e-30)


def animate_collapse_disk(result: SimulationResult, output_path: str | Path) -> Path:
    """Animazione ALTERNATIVA, puramente estetica (nessuna nuova fisica,
    nessun nuovo calcolo dinamico): la stella disegnata come un disco 2D
    pieno, colorato per densita' locale di shell (scala log), che si
    restringe nel tempo usando ESATTAMENTE i raggi di shell gia' calcolati
    da `collasso.dynamics` (`CollapseSolution.r_cm`) — l'unica grandezza
    derivata in piu' e' la densita' per shell per il colore (vedi
    `_shell_densities_gcm3`), usata SOLO per la resa visiva.

    Si AFFIANCA a `animate_collapse` (che resta invariata, stesso
    contenuto informativo di prima): file di output distinto, non una
    sostituzione.

    Implementazione: ogni shell e' un cerchio pieno concentrico, disegnato
    dalla shell PIU' ESTERNA (raggio maggiore) verso quella PIU' INTERNA
    (raggio minore, zorder crescente) — i cerchi piu' piccoli, disegnati
    sopra, producono l'effetto visivo di anelli concentrici senza dover
    costruire vere corone circolari.

    L'ULTIMO fotogramma include un testo esplicito sovrapposto che
    dichiara la fine della simulazione e l'assenza di un bounce modellato
    — lo stesso limite di modello gia' dichiarato nei disclaimer
    (`collasso.pipeline._build_disclaimers`, voce 4), qui reso
    visivamente esplicito, non un'informazione nuova.

    Crea le directory intermedie di `output_path` se non esistono. Ritorna
    il path assoluto scritto.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sol = result.collapse_solution
    delta_m_g = result.delta_m_g
    n_shell = sol.r_cm.shape[0]
    indici = _frame_indices(len(sol.t_s))

    # Densita' per shell per OGNI frame campionato (solo per il colore) —
    # calcolate una volta sola qui, non ad ogni chiamata di _update.
    densita_per_frame = [_shell_densities_gcm3(sol.r_cm[:, i], delta_m_g) for i in indici]
    rho_min = min(float(np.min(d)) for d in densita_per_frame)
    rho_max = max(float(np.max(d)) for d in densita_per_frame)
    # Normalizzazione FISSA su tutta l'animazione (non per-frame): un colore
    # deve sempre rappresentare la stessa densita' in ogni fotogramma,
    # altrimenti il colore cambierebbe solo per effetto della rinormalizzazione,
    # non per una vera variazione fisica di densita'.
    norm = LogNorm(vmin=max(rho_min, 1.0), vmax=max(rho_max, rho_min + 1.0))
    cmap = plt.get_cmap("inferno")

    r_max_km = float(np.max(sol.r_cm[-1, :]) / 1.0e5) * 1.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(-r_max_km, r_max_km)
    ax.set_ylim(-r_max_km, r_max_km)
    ax.set_xlabel("km")
    ax.set_ylabel("km")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Densita' di shell (g/cm^3)")
    fig.tight_layout()

    cerchi: list[Circle] = []
    testo_finale = {"artista": None}

    def _update(frame_idx: int):
        for cerchio in cerchi:
            cerchio.remove()
        cerchi.clear()
        if testo_finale["artista"] is not None:
            testo_finale["artista"].remove()
            testo_finale["artista"] = None

        i = indici[frame_idx]
        r_km = sol.r_cm[:, i] / 1.0e5
        densita = densita_per_frame[frame_idx]

        for k in range(n_shell - 1, -1, -1):
            colore = cmap(norm(densita[k]))
            # zorder crescente al decrescere di k: la shell piu' interna
            # (k=0, raggio minore) va disegnata per ultima/sopra le altre.
            cerchio = Circle(
                (0.0, 0.0), r_km[k], facecolor=colore, edgecolor="none", zorder=n_shell - 1 - k
            )
            ax.add_patch(cerchio)
            cerchi.append(cerchio)

        t_ms = sol.t_s[i] * 1.0e3
        rho_c = sol.rho_c_gcm3_t[i]
        ax.set_title(
            f"Collasso di {result.star.id} (vista 2D): t={t_ms:.4f} ms, "
            f"rho_c={rho_c:.3e} g/cm^3"
        )

        if frame_idx == len(indici) - 1:
            testo_finale["artista"] = ax.text(
                0.5,
                0.03,
                TESTO_FINE_SIMULAZIONE,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
                color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.75),
                zorder=n_shell + 1,
            )

        return cerchi

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
