"""
scripts/run_simulation.py — ENTRY POINT UNICO della simulazione del collasso
gravitazionale stellare (Step 7).

Uso:
    python scripts/run_simulation.py <star_id>   # non interattivo
    python scripts/run_simulation.py             # interattivo, chiede l'id

Compone `collasso.pipeline.run_full_simulation` (output numerico completo:
profilo iniziale, dinamica, energie, classificazione remnant) con
`collasso.visualization` (animazione a linea + animazione 2D a disco +
grafico riassuntivo PNG), stampando SEMPRE la sezione "LIMITI DEL MODELLO"
con tutti gli 8 (o 9, per le voci proxy) disclaimer (regola CLAUDE.md:
nessun log grezzo di solve_ivp, solo output strutturato filtrato).

Formattazione con `rich` (ciclo successivo, puramente estetico — vedi
STATUS.md): stesso identico contenuto numerico di prima, nessuna cifra
omessa o arrotondata diversamente, solo presentazione a pannelli/tabelle
invece di testo grezzo con separatori "=".
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collasso.catalog import load_reference_catalog  # noqa: E402
from collasso.pipeline import run_full_simulation  # noqa: E402
from collasso.visualization import animate_collapse, animate_collapse_disk, plot_summary  # noqa: E402

MAX_TENTATIVI_INPUT = 3

# Sottocartella di output (GIF + PNG), creata se assente, dentro la root del
# progetto (parent di scripts/).
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

console = Console()


def _seleziona_star_id() -> str:
    """Determina lo `star_id` da usare: `sys.argv[1]` se presente (uso non
    interattivo/test), altrimenti chiede interattivamente all'utente,
    validando contro il catalogo disponibile (max MAX_TENTATIVI_INPUT
    tentativi, poi errore chiaro).
    """
    if len(sys.argv) > 1:
        return sys.argv[1]

    catalogo = load_reference_catalog()
    id_disponibili = [p.id for p in catalogo]
    console.print(f"Stelle disponibili nel catalogo di riferimento: [bold]{id_disponibili}[/bold]")

    for tentativo in range(1, MAX_TENTATIVI_INPUT + 1):
        scelta = input("Seleziona una stella: ").strip()
        if scelta in id_disponibili:
            return scelta
        console.print(
            f"[yellow]Id '{scelta}' non valido (tentativo {tentativo}/{MAX_TENTATIVI_INPUT}).[/yellow] "
            f"Id disponibili: {id_disponibili}"
        )

    raise ValueError(
        f"Nessun id valido fornito dopo {MAX_TENTATIVI_INPUT} tentativi. "
        f"Id disponibili: {id_disponibili}"
    )


def _tabella_kv(righe: list[tuple[str, str]]) -> Table:
    """Tabella chiave/valore compatta (senza intestazione, senza bordi
    pesanti) — stile riusato da ogni sezione dell'output per uniformita'.
    """
    tabella = Table(show_header=False, box=None, padding=(0, 1, 0, 0))
    tabella.add_column(style="bold cyan", no_wrap=True)
    tabella.add_column()
    for chiave, valore in righe:
        tabella.add_row(chiave, valore)
    return tabella


def _stampa_risultato(result) -> None:
    """Stampa l'output numerico strutturato completo (sezioni: Stella,
    Profilo iniziale, Dinamica, Energie, Classificazione remnant, Limiti
    del modello). Nessun log grezzo di solve_ivp (regola CLAUDE.md).
    """
    star = result.star
    profile = result.profile
    sol = result.collapse_solution
    energie = result.energies

    console.print(
        Panel(
            f"[bold]SIMULAZIONE COMPLETA DEL COLLASSO[/bold] - stella [bold yellow]'{star.id}'[/bold yellow]",
            expand=False,
            border_style="bold blue",
        )
    )

    # --- Sezione: Stella --------------------------------------------------
    console.print("\n[bold underline]Stella[/bold underline] (catalogo di riferimento, Step 1)")
    console.print(
        _tabella_kv(
            [
                ("id", star.id),
                ("massa_zams_msun", f"{star.massa_zams_msun:.2f}"),
                ("massa_nucleo_msun", f"{star.massa_nucleo_msun:.4f}"),
                ("densita_centrale_gcm3", f"{star.densita_centrale_gcm3:.3e}"),
                ("raggio_iniziale_km", f"{star.raggio_iniziale_km:.1f}"),
                ("ye", f"{star.ye:.2f}"),
                ("n_politropico", f"{star.n_politropico:.2f}"),
            ]
        )
    )
    console.print(f"[dim]fonte:[/dim] {star.fonte}")

    # --- Sezione: Profilo iniziale ----------------------------------------
    console.print("\n[bold underline]Profilo di equilibrio iniziale[/bold underline] (Lane-Emden, Step 2)")
    scarto_pct = 100.0 * abs(profile.R_km - star.raggio_iniziale_km) / star.raggio_iniziale_km
    console.print(
        _tabella_kv(
            [
                ("xi1", f"{result.lane_emden_solution.xi1:.5f}"),
                ("R_derivato_km", f"{profile.R_km:.2f}"),
                ("raggio_iniziale_km (catalogo)", f"{star.raggio_iniziale_km:.1f}"),
                ("scarto raggio", f"{scarto_pct:.6f}% (atteso ~0, vedi fonte)"),
                ("scarto K", f"[bold]{result.k_deviation_pct:+.2f}%[/bold] (vs. EOS esatta - vedi disclaimer 2)"),
                ("N_shell", str(len(result.delta_m_g))),
                ("t_ff", f"{result.t_ff_s * 1.0e3:.4f} ms"),
                ("t_max", f"{result.t_max_s * 1.0e3:.4f} ms"),
            ]
        )
    )

    # --- Sezione: Dinamica (con gestione esplicita di collapsed=False) ----
    console.print("\n[bold underline]Dinamica del collasso[/bold underline] (shell Lagrangiane, Step 4/5, relativistic=True)")
    stato_collasso = "[bold green]True[/bold green]" if sol.collapsed else "[bold red]False[/bold red]"
    console.print(
        _tabella_kv(
            [
                ("collapsed", stato_collasso),
                ("collapse_reason", str(sol.collapse_reason)),
                ("t_collapse_s", str(sol.t_collapse_s)),
                ("rho_c iniziale", f"{sol.rho_c_gcm3_t[0]:.4e} g/cm^3"),
                ("rho_c finale (ultimo istante)", f"{sol.rho_c_gcm3_t[-1]:.4e} g/cm^3"),
            ]
        )
    )
    if sol.collapsed:
        console.print(
            f"[green]Collasso avvenuto a t={sol.t_collapse_s * 1.0e3:.4f} ms "
            f"(motivo: {sol.collapse_reason}).[/green]"
        )
    else:
        console.print(
            "[yellow]Collasso NON completato entro t_max_s[/yellow] - l'integrazione si e' "
            "fermata al tempo massimo senza che nessun evento fisico scattasse; le "
            "energie/densita' riportate sotto sono all'ULTIMO istante disponibile "
            f"(t={energie.t_energie_s * 1.0e3:.4f} ms), non a un vero momento di collasso."
        )

    # --- Sezione: Energie ----------------------------------------------------
    console.print("\n[bold underline]Bilancio energetico discreto[/bold underline] (Step 7, approssimazione non validata)")
    console.print(
        _tabella_kv(
            [
                ("KE_finale_erg", f"{energie.KE_finale_erg:.4e} (t={energie.t_energie_s * 1.0e3:.4f} ms)"),
                ("PE_iniziale_erg", f"{energie.PE_iniziale_erg:.4e}"),
                ("PE_finale_erg", f"{energie.PE_finale_erg:.4e}"),
                ("Delta_PE_erg", f"{energie.Delta_PE_erg:.4e} (atteso negativo)"),
            ]
        )
    )
    console.print(
        "[dim]NOTA: il bilancio include SOLO cinetica + autoenergia gravitazionale "
        "discretizzata (massa a meta' shell) - NON energia interna/termica ne' "
        "perdite per neutrini; Delta_PE_erg + KE_finale_erg NON e' una verifica di "
        "conservazione dell'energia (bilancio non chiuso per costruzione).[/dim]"
    )

    # --- Sezione: Classificazione remnant --------------------------------
    console.print("\n[bold underline]Classificazione del remnant[/bold underline] (Step 6/Retrofit)")
    console.print(f"M_Chandrasekhar(ye={star.ye}) = [bold]{result.m_chandrasekhar_msun:.4f} Msun[/bold] (Newtoniana classica, Step 3)")

    tabella_remnant = Table(box=None)
    tabella_remnant.add_column("EOS", style="bold cyan")
    tabella_remnant.add_column("M_TOV (Msun)")
    tabella_remnant.add_column("rho_c al massimo (g/cm^3)")
    tabella_remnant.add_column("Classificazione", style="bold")
    tabella_remnant.add_row("SLy", f"{result.m_tov_sly_msun:.4f}", f"{result.rho_c_tov_sly_gcm3:.3e}", result.remnant_class_sly)
    tabella_remnant.add_row("APR4", f"{result.m_tov_apr4_msun:.4f}", f"{result.rho_c_tov_apr4_gcm3:.3e}", result.remnant_class_apr4)
    console.print(tabella_remnant)

    # --- Sezione: Limiti del modello (sempre visibile, 8/9 disclaimer) ---
    testo_disclaimer = "\n\n".join(f"[bold yellow]{i}.[/bold yellow] {d}" for i, d in enumerate(result.disclaimers, start=1))
    console.print()
    console.print(
        Panel(
            testo_disclaimer,
            title="[bold]LIMITI DEL MODELLO[/bold] (disclaimer, sempre mostrati)",
            border_style="yellow",
            expand=True,
        )
    )


def main() -> None:
    star_id = _seleziona_star_id()
    result = run_full_simulation(star_id)

    _stampa_risultato(result)

    # --- Visualizzazione (Step 7 + animazione 2D a disco, ciclo successivo) -
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gif_path = animate_collapse(result, OUTPUT_DIR / f"collapse_{star_id}.gif")
    disk_gif_path = animate_collapse_disk(result, OUTPUT_DIR / f"collapse_disk_{star_id}.gif")
    png_path = plot_summary(result, OUTPUT_DIR / f"summary_{star_id}.png")

    console.print("\n[bold underline]File generati[/bold underline]")
    console.print(f"Animazione (raggio vs massa): [link]{gif_path}[/link]")
    console.print(f"Animazione 2D (disco colorato per densita'): [link]{disk_gif_path}[/link]")
    console.print(f"Grafico riassuntivo: [link]{png_path}[/link]")


if __name__ == "__main__":
    main()
