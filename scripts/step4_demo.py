"""
scripts/step4_demo.py — demo Step 4: dinamica shell Lagrangiane del collasso
gravitazionale per la stella scelta s20.

Contenuti (vedi piano Step 4, STATUS.md, sezione "2. Script demo"):
- confronto massa_nucleo_msun (s20) vs chandrasekhar_mass_msun(0.46);
- costruzione della condizione iniziale (Lane-Emden n=2.0 -> physical_profile
  -> build_initial_shells, N=N_SHELLS_DEMO_DEFAULT);
- simulazione della dinamica (simulate_collapse) e stampa delle metriche
  (densita' centrale iniziale/finale, collapsed, t_collapse_s,
  collapse_reason);
- disclaimer obbligatorio sui limiti dichiarati di questo step.

Nessun plot Matplotlib (riservato a Step 7). Output filtrato (regola
CLAUDE.md): solo le righe di riepilogo elencate sopra, nessun log grezzo.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collasso.catalog import get_progenitor_by_id, load_reference_catalog
from collasso.dynamics import T_MAX_FREE_FALL_MULTIPLIER, build_initial_shells, free_fall_time_s, simulate_collapse
from collasso.eos import chandrasekhar_mass_msun
from collasso.lane_emden import physical_profile, solve_lane_emden

# Numero di shell usato in questo script demo (bilancio risoluzione/tempo di
# esecuzione: sistema ODE 2*200=400 componenti). Fissato dal planner,
# STATUS.md, piano Step 4 REV. 2.
N_SHELLS_DEMO_DEFAULT = 200


def main() -> None:
    catalogo = load_reference_catalog()
    stella = get_progenitor_by_id(catalogo, "s20")

    # --- 2.2: confronto massa nucleo vs massa di Chandrasekhar -------------
    m_ch = chandrasekhar_mass_msun(stella.ye)
    scarto_pct = 100.0 * (stella.massa_nucleo_msun - m_ch) / m_ch
    print("=== Step 4 demo: dinamica del collasso, stella scelta s20 ===")
    print(
        f"massa_nucleo_msun={stella.massa_nucleo_msun:.4f} Msun vs "
        f"chandrasekhar_mass_msun(ye={stella.ye})={m_ch:.4f} Msun "
        f"-> scarto={scarto_pct:+.2f}% "
        f"({'SUPRA' if stella.massa_nucleo_msun > m_ch else 'SUB'}-Chandrasekhar)"
    )

    # --- 2.3: condizione iniziale (Lane-Emden -> physical_profile -> shell) -
    solution = solve_lane_emden(n=stella.n_politropico)
    profile = physical_profile(solution, stella.densita_centrale_gcm3, stella.massa_nucleo_msun)
    delta_m_g, m_enclosed_g, r0_cm = build_initial_shells(
        profile.r_cm, profile.rho_gcm3, N_SHELLS_DEMO_DEFAULT
    )
    print(
        f"Condizione iniziale: N_shell={N_SHELLS_DEMO_DEFAULT}, "
        f"R_iniziale={r0_cm[-1] / 1.0e5:.1f} km, "
        f"rho_c_iniziale={stella.densita_centrale_gcm3:.3e} g/cm^3"
    )

    # --- 2.4: scala temporale -----------------------------------------------
    t_ff_s = free_fall_time_s(stella.densita_centrale_gcm3)
    t_max_s = T_MAX_FREE_FALL_MULTIPLIER * t_ff_s
    print(f"Tempo di caduta libera t_ff={t_ff_s * 1.0e3:.3f} ms; t_max={t_max_s * 1.0e3:.3f} ms (=10*t_ff)")

    # --- 2.5: simulazione ----------------------------------------------------
    risultato = simulate_collapse(delta_m_g, r0_cm, ye=stella.ye, t_max_s=t_max_s)

    # --- 2.6: metriche --------------------------------------------------------
    rho_c_iniziale = risultato.rho_c_gcm3_t[0]
    rho_c_finale = risultato.rho_c_gcm3_t[-1]
    print(f"Densita' centrale (shell 1): iniziale={rho_c_iniziale:.4e} g/cm^3, finale={rho_c_finale:.4e} g/cm^3")
    print(f"collapsed={risultato.collapsed}")
    if risultato.collapsed:
        rapporto_tmax = risultato.t_collapse_s / t_max_s
        rapporto_tff = risultato.t_collapse_s / t_ff_s
        print(
            f"t_collapse={risultato.t_collapse_s * 1.0e3:.4f} ms "
            f"(={rapporto_tmax:.4f}*t_max, ={rapporto_tff:.3f}*t_ff); "
            f"collapse_reason='{risultato.collapse_reason}'"
        )
    else:
        print("t_collapse_s=None, collapse_reason=None (nessun collasso entro t_max)")

    # --- 2.7: disclaimer obbligatorio -----------------------------------------
    print(
        "DISCLAIMER (limiti di modello dichiarati, Step 4): gravita' SOLO "
        "Newtoniana (nessuna correzione relativistica sul nucleo, Step 5); "
        "nessun termine di raffreddamento neutrini; nessuna viscosita' "
        "artificiale (possibile ringing numerico in forte compressione); "
        "condizione iniziale NON esattamente in equilibrio idrostatico "
        "rispetto alla EOS reale (mismatch noto del catalogo placeholder, "
        "Step 1/2); integrazione fermata all'innesco del collasso "
        "(soglia r_min_frac o shell crossing), non al bounce."
    )


if __name__ == "__main__":
    main()
