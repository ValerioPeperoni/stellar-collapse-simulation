"""
scripts/run_simulation.py — ENTRY POINT UNICO della simulazione del collasso
gravitazionale stellare (Step 7).

Uso:
    python scripts/run_simulation.py <star_id>   # non interattivo
    python scripts/run_simulation.py             # interattivo, chiede l'id

Compone `collasso.pipeline.run_full_simulation` (output numerico completo:
profilo iniziale, dinamica, energie, classificazione remnant) con
`collasso.visualization` (animazione GIF + grafico riassuntivo PNG),
stampando SEMPRE la sezione "LIMITI DEL MODELLO" con tutti gli 8 disclaimer
(regola CLAUDE.md: nessun log grezzo di solve_ivp, solo output strutturato
filtrato).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collasso.catalog import load_reference_catalog  # noqa: E402
from collasso.pipeline import run_full_simulation  # noqa: E402
from collasso.visualization import animate_collapse, plot_summary  # noqa: E402

MAX_TENTATIVI_INPUT = 3

# Sottocartella di output (GIF + PNG), creata se assente, dentro la root del
# progetto (parent di scripts/).
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


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
    print(f"Stelle disponibili nel catalogo di riferimento: {id_disponibili}")

    for tentativo in range(1, MAX_TENTATIVI_INPUT + 1):
        scelta = input("Seleziona una stella: ").strip()
        if scelta in id_disponibili:
            return scelta
        print(
            f"Id '{scelta}' non valido (tentativo {tentativo}/{MAX_TENTATIVI_INPUT}). "
            f"Id disponibili: {id_disponibili}"
        )

    raise ValueError(
        f"Nessun id valido fornito dopo {MAX_TENTATIVI_INPUT} tentativi. "
        f"Id disponibili: {id_disponibili}"
    )


def _stampa_risultato(result) -> None:
    """Stampa l'output numerico strutturato completo (sezioni: Stella,
    Profilo iniziale, Dinamica, Energie, Classificazione remnant, Limiti
    del modello). Nessun log grezzo di solve_ivp (regola CLAUDE.md).
    """
    star = result.star
    profile = result.profile
    sol = result.collapse_solution
    energie = result.energies

    print("=" * 78)
    print(f"SIMULAZIONE COMPLETA DEL COLLASSO - stella '{star.id}'")
    print("=" * 78)

    # --- Sezione: Stella --------------------------------------------------
    print("\n--- Stella (catalogo di riferimento, Step 1) ---")
    print(
        f"id={star.id}, massa_zams_msun={star.massa_zams_msun:.2f}, "
        f"massa_nucleo_msun={star.massa_nucleo_msun:.4f}, "
        f"densita_centrale_gcm3={star.densita_centrale_gcm3:.3e}, "
        f"raggio_iniziale_km={star.raggio_iniziale_km:.1f}, ye={star.ye:.2f}, "
        f"n_politropico={star.n_politropico:.2f}"
    )
    print(f"fonte: {star.fonte}")

    # --- Sezione: Profilo iniziale ----------------------------------------
    print("\n--- Profilo di equilibrio iniziale (Lane-Emden, Step 2) ---")
    scarto_pct = 100.0 * abs(profile.R_km - star.raggio_iniziale_km) / star.raggio_iniziale_km
    print(
        f"xi1={result.lane_emden_solution.xi1:.5f}, R_derivato_km={profile.R_km:.2f}, "
        f"raggio_iniziale_km (catalogo)={star.raggio_iniziale_km:.1f}, "
        f"scarto={scarto_pct:.6f}% (atteso ~0: raggio_iniziale_km del catalogo "
        "e' gia' derivato con questa stessa formula, vedi fonte)"
    )
    print(
        f"Scarto K (costante politropica implicita vs. EOS di Chandrasekhar "
        f"esatta alla stessa densita' centrale): {result.k_deviation_pct:+.2f}% "
        "- misura diretta di quanto questo nucleo sia supra-Chandrasekhar "
        "(NON un errore di modellazione, vedi disclaimer 2 sotto)."
    )
    print(f"N_shell={len(result.delta_m_g)}, t_ff={result.t_ff_s * 1.0e3:.4f} ms, t_max={result.t_max_s * 1.0e3:.4f} ms")

    # --- Sezione: Dinamica (con gestione esplicita di collapsed=False) ----
    print("\n--- Dinamica del collasso (shell Lagrangiane, Step 4/5, relativistic=True) ---")
    print(
        f"collapsed={sol.collapsed}, collapse_reason={sol.collapse_reason}, "
        f"t_collapse_s={sol.t_collapse_s}"
    )
    if sol.collapsed:
        print(
            f"Collasso avvenuto a t={sol.t_collapse_s * 1.0e3:.4f} ms "
            f"(motivo: {sol.collapse_reason})."
        )
    else:
        print(
            "Collasso NON completato entro t_max_s - l'integrazione si e' "
            "fermata al tempo massimo senza che nessun evento fisico "
            "scattasse; le energie/densita' riportate sotto sono "
            "all'ULTIMO istante disponibile (t="
            f"{energie.t_energie_s * 1.0e3:.4f} ms), non a un vero momento "
            "di collasso."
        )
    print(
        f"Densita' centrale (shell 1): iniziale={sol.rho_c_gcm3_t[0]:.4e} g/cm^3, "
        f"finale (ultimo istante disponibile)={sol.rho_c_gcm3_t[-1]:.4e} g/cm^3"
    )

    # --- Sezione: Energie ----------------------------------------------------
    print("\n--- Bilancio energetico discreto (Step 7, approssimazione non validata) ---")
    print(
        f"KE_finale_erg={energie.KE_finale_erg:.4e} (all'ultimo istante disponibile, "
        f"t={energie.t_energie_s * 1.0e3:.4f} ms)"
    )
    print(
        f"PE_iniziale_erg={energie.PE_iniziale_erg:.4e}, "
        f"PE_finale_erg={energie.PE_finale_erg:.4e}, "
        f"Delta_PE_erg={energie.Delta_PE_erg:.4e} (atteso negativo: energia liberata)"
    )
    print(
        "NOTA: il bilancio include SOLO cinetica + autoenergia gravitazionale "
        "discretizzata (massa a meta' shell) - NON energia interna/termica ne' "
        "perdite per neutrini; Delta_PE_erg + KE_finale_erg NON e' una "
        "verifica di conservazione dell'energia (bilancio non chiuso per "
        "costruzione)."
    )

    # --- Sezione: Classificazione remnant --------------------------------
    print("\n--- Classificazione del remnant (Step 6/Retrofit) ---")
    print(f"M_Chandrasekhar(ye={star.ye})={result.m_chandrasekhar_msun:.4f} Msun (Newtoniana classica, Step 3)")
    print(
        f"SLy:  M_TOV={result.m_tov_sly_msun:.4f} Msun (rho_c={result.rho_c_tov_sly_gcm3:.3e} g/cm^3) "
        f"-> classificazione='{result.remnant_class_sly}'"
    )
    print(
        f"APR4: M_TOV={result.m_tov_apr4_msun:.4f} Msun (rho_c={result.rho_c_tov_apr4_gcm3:.3e} g/cm^3) "
        f"-> classificazione='{result.remnant_class_apr4}'"
    )

    # --- Sezione: Limiti del modello (sempre visibile, 8 disclaimer) ------
    print("\n" + "=" * 78)
    print("LIMITI DEL MODELLO (disclaimer, sempre mostrati)")
    print("=" * 78)
    for i, disclaimer in enumerate(result.disclaimers, start=1):
        print(f"{i}. {disclaimer}")


def main() -> None:
    star_id = _seleziona_star_id()
    result = run_full_simulation(star_id)

    _stampa_risultato(result)

    # --- Visualizzazione (Step 7) ------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gif_path = animate_collapse(result, OUTPUT_DIR / f"collapse_{star_id}.gif")
    png_path = plot_summary(result, OUTPUT_DIR / f"summary_{star_id}.png")

    print("\n--- File generati ---")
    print(f"Animazione: {gif_path}")
    print(f"Grafico riassuntivo: {png_path}")


if __name__ == "__main__":
    main()
