"""
scripts/step5_demo.py — demo Step 5: correzioni relativistiche sul nucleo
per la stella scelta s20.

Contenuti (vedi piano Step 5, STATUS.md, sezione "4. scripts/step5_demo.py"):
- riusa lo stesso scenario di Step 4 (stella s20, catalogo di riferimento);
- esegue simulate_collapse due volte: relativistic=False (baseline, deve
  riprodurre i numeri gia' noti di Step 4) e relativistic=True;
- stampa: tempo di collasso in entrambi i casi, compattezza massima
  raggiunta (shell piu' interna) nel caso relativistico, fattore di
  correzione massimo incontrato all'istante finale;
- disclaimer obbligatorio sui limiti dichiarati di questo step.

Nessun plot Matplotlib (riservato a Step 7). Output filtrato (regola
CLAUDE.md): solo le righe di riepilogo elencate sopra, nessun log grezzo.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collasso.catalog import get_progenitor_by_id, load_reference_catalog
from collasso.constants import C_LIGHT_CGS, G_CGS
from collasso.dynamics import T_MAX_FREE_FALL_MULTIPLIER, build_initial_shells, free_fall_time_s, simulate_collapse
from collasso.eos import pressure_chandrasekhar
from collasso.lane_emden import physical_profile, solve_lane_emden
from collasso.relativistic import relativistic_grav_accel

# Stesso numero di shell dello script demo di Step 4 (STATUS.md, piano
# Step 4 REV. 2).
N_SHELLS_DEMO_DEFAULT = 200


def main() -> None:
    catalogo = load_reference_catalog()
    stella = get_progenitor_by_id(catalogo, "s20")

    # --- Condizione iniziale (identica a Step 4) ----------------------------
    solution = solve_lane_emden(n=stella.n_politropico)
    profile = physical_profile(solution, stella.densita_centrale_gcm3, stella.massa_nucleo_msun)
    delta_m_g, m_enclosed_g, r0_cm = build_initial_shells(profile.r_cm, profile.rho_gcm3, N_SHELLS_DEMO_DEFAULT)

    t_ff_s = free_fall_time_s(stella.densita_centrale_gcm3)
    t_max_s = T_MAX_FREE_FALL_MULTIPLIER * t_ff_s

    print("=== Step 5 demo: correzioni relativistiche sul nucleo, stella scelta s20 ===")
    print(
        f"Condizione iniziale: N_shell={N_SHELLS_DEMO_DEFAULT}, "
        f"R_iniziale={r0_cm[-1] / 1.0e5:.1f} km, "
        f"rho_c_iniziale={stella.densita_centrale_gcm3:.3e} g/cm^3, "
        f"t_max={t_max_s * 1.0e3:.3f} ms (=10*t_ff)"
    )

    # --- Simulazione Newtoniana (baseline, Step 4) --------------------------
    risultato_newt = simulate_collapse(delta_m_g, r0_cm, ye=stella.ye, t_max_s=t_max_s, relativistic=False)
    print(
        f"[relativistic=False] collapsed={risultato_newt.collapsed}, "
        f"t_collapse={risultato_newt.t_collapse_s * 1.0e3:.4f} ms, "
        f"collapse_reason='{risultato_newt.collapse_reason}'"
    )

    # --- Simulazione con correzione relativistica ----------------------------
    risultato_gr = simulate_collapse(delta_m_g, r0_cm, ye=stella.ye, t_max_s=t_max_s, relativistic=True)
    print(
        f"[relativistic=True]  collapsed={risultato_gr.collapsed}, "
        f"t_collapse={risultato_gr.t_collapse_s * 1.0e3:.4f} ms, "
        f"collapse_reason='{risultato_gr.collapse_reason}'"
    )

    # --- Confronto tempi di collasso ------------------------------------------
    if risultato_newt.t_collapse_s is not None and risultato_gr.t_collapse_s is not None:
        delta_t_pct = 100.0 * (risultato_gr.t_collapse_s - risultato_newt.t_collapse_s) / risultato_newt.t_collapse_s
        direzione = "PIU RAPIDO" if delta_t_pct < 0 else "PIU LENTO"
        print(
            f"Confronto: t_collapse relativistico {direzione} "
            f"del {abs(delta_t_pct):.2f}% rispetto al caso Newtoniano "
            f"(atteso: piu' rapido, gravita' effettiva piu' forte)"
        )

    # --- Compattezza massima e fattore di correzione massimo, caso GR --------
    r_finale_cm = risultato_gr.r_cm[:, -1]
    m_enclosed_finale_g = m_enclosed_g  # massa di shell fissa (Lagrangiana), non cambia nel tempo
    compattezza_shell_finali = 2.0 * G_CGS * m_enclosed_finale_g / (r_finale_cm * C_LIGHT_CGS**2)
    compattezza_max = compattezza_shell_finali.max()
    idx_max = int(compattezza_shell_finali.argmax())

    # Fattore di correzione alla shell piu' interna (indice 0), istante finale.
    r0_finale = r_finale_cm[0]
    m0_g = m_enclosed_finale_g[0]
    v_shell0 = (4.0 / 3.0) * 3.141592653589793 * r0_finale**3
    rho0_finale = delta_m_g[0] / max(v_shell0, 1e-30)
    p0_finale = pressure_chandrasekhar(rho0_finale, stella.ye)
    a_grav_gr_0 = relativistic_grav_accel(m0_g, r0_finale, p0_finale, rho0_finale)
    a_grav_newt_0 = -G_CGS * m0_g / r0_finale**2
    # Nota naming (segnalato dal critic-fisico): questo NON e' il fattore
    # massimo su tutte le shell (quello si trova all'indice di compattezza
    # massima, idx_max, tipicamente diverso da 0) — e' specificamente il
    # fattore alla shell PIU' INTERNA (indice 0), etichettato di conseguenza.
    fattore_correzione_shell_interna = abs(a_grav_gr_0 / a_grav_newt_0)

    print(
        f"Compattezza massima raggiunta (caso GR, istante finale, su tutte le shell)="
        f"{compattezza_max:.4e} (shell indice {idx_max}); "
        f"fattore di correzione a_grav_GR/a_grav_Newt alla shell piu' interna (indice 0)="
        f"{fattore_correzione_shell_interna:.4f}"
    )

    # --- Disclaimer obbligatorio ----------------------------------------------
    print(
        "DISCLAIMER (limiti di modello dichiarati, Step 5): potenziale "
        "gravitazionale effettivo APPROSSIMATO ('Case A', Marek, Dimmelmeier, "
        "Janka, Mueller & Buras 2006), NON idrodinamica GR completa; il "
        "termine di pressione (gradiente idrostatico) NON e' corretto "
        "relativisticamente, solo il termine sorgente di gravita'; nessun "
        "limite TOV/classificazione remnant (Step 6); nessun termine di "
        "raffreddamento neutrini; nessuna rotazione/campo magnetico "
        "(limiti gia' esclusi a monte)."
    )


if __name__ == "__main__":
    main()
