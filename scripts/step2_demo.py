"""
scripts/step2_demo.py — demo Step 2: validazione del solver di Lane-Emden
contro soluzioni analitiche note (n=0,1,5) e profilo fisico della stella
scelta del catalogo di riferimento (s20).

Output filtrato (regola CLAUDE.md): solo le righe di riepilogo dei
confronti, nessun log grezzo di integrazione.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Permette di eseguire lo script direttamente (python scripts/step2_demo.py)
# senza dover installare il pacchetto o impostare PYTHONPATH a mano (stesso
# pattern usato in scripts/step1_demo.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collasso.catalog import get_progenitor_by_id, load_reference_catalog
from collasso.lane_emden import physical_profile, solve_lane_emden, theta_analytic, xi1_analytic

# Tolleranze di confronto (stesse fissate per i test, punto 3.1 del piano
# Step 2 — usate qui per coerenza demo/test).
TOL_THETA_RTOL = 1e-6
TOL_THETA_ATOL = 1e-8
TOL_THETA_N5_RTOL = 1e-5
TOL_THETA_N5_ATOL = 1e-7


def _compute_theta_errors(theta_num: np.ndarray, theta_an: np.ndarray) -> tuple[float, float]:
    """Calcola scarto assoluto e relativo tra soluzione numerica e analitica.

    Il relativo viene calcolato solo dove theta_an != 0; dove theta_an == 0,
    usa un massimo sicuro per evitare divisioni fuorvianti.

    Ritorna:
        (scarto_assoluto_max, scarto_relativo_max)
    """
    scarto_assoluto = np.max(np.abs(theta_num - theta_an))
    with np.errstate(divide="ignore", invalid="ignore"):
        scarto_relativo = np.max(
            np.abs(
                (theta_num - theta_an) / np.where(theta_an != 0, theta_an, 1.0)
            )
        )
    return scarto_assoluto, scarto_relativo


def _validate_against_analytic(n: int, xi1_an: float) -> None:
    """Valida solve_lane_emden(n) contro theta_analytic/xi1_analytic (n=0,1).

    Nota: per n=0,1, stampa solo lo scarto assoluto (il relativo e' fuorviante
    vicino a xi1 dove theta_an->0). Il vero criterio di conformita' e'
    np.allclose (linea stampa).
    """
    solution = solve_lane_emden(n)
    theta_an = theta_analytic(solution.xi, n)

    scarto_assoluto, _ = _compute_theta_errors(solution.theta, theta_an)
    entro_tolleranza = np.allclose(
        solution.theta, theta_an, rtol=TOL_THETA_RTOL, atol=TOL_THETA_ATOL
    )
    scarto_xi1 = abs(solution.xi1 - xi1_an)

    print(
        f"n={n}: scarto_theta_max_abs={scarto_assoluto:.3e} "
        f"entro_tolleranza={entro_tolleranza} "
        f"xi1_num={solution.xi1:.6f} xi1_analitico={xi1_an:.6f} scarto_xi1={scarto_xi1:.3e}"
    )


def _validate_n5() -> None:
    """Valida solve_lane_emden(5, xi_max=50.0) contro (1+xi^2/3)^-1/2 inline."""
    solution = solve_lane_emden(5, xi_max=50.0)
    theta_an = (1.0 + solution.xi**2 / 3.0) ** (-0.5)

    scarto_assoluto, scarto_relativo = _compute_theta_errors(solution.theta, theta_an)
    entro_tolleranza = np.allclose(
        solution.theta, theta_an, rtol=TOL_THETA_N5_RTOL, atol=TOL_THETA_N5_ATOL
    )

    print(
        f"n=5: scarto_theta_max_abs={scarto_assoluto:.3e} "
        f"scarto_theta_max_rel={scarto_relativo:.3e} entro_tolleranza={entro_tolleranza}"
    )
    print("n=5: xi1 non definito per n=5 (nessuna superficie finita entro xi_max=50, atteso)")
    assert solution.xi1 is None
    assert solution.dtheta_dxi_at_xi1 is None


def _physical_profile_stella_scelta() -> None:
    """Profilo fisico della stella scelta (s20) e cross-check col catalogo."""
    catalogo = load_reference_catalog()
    stella = get_progenitor_by_id(catalogo, "s20")

    solution = solve_lane_emden(n=stella.n_politropico)
    profilo = physical_profile(
        solution, stella.densita_centrale_gcm3, stella.massa_nucleo_msun
    )

    scarto_percentuale = (
        100.0
        * abs(profilo.R_km - stella.raggio_iniziale_km)
        / stella.raggio_iniziale_km
    )

    print(
        f"Stella scelta: id={stella.id} n={stella.n_politropico} "
        f"rho_c_gcm3={stella.densita_centrale_gcm3:.3e} "
        f"massa_nucleo_msun={stella.massa_nucleo_msun}"
    )
    print(
        f"R_derivato_km={profilo.R_km:.3f} "
        f"raggio_iniziale_km(catalogo)={stella.raggio_iniziale_km:.3f} "
        f"scarto_percentuale={scarto_percentuale:.1f}%"
    )
    print(
        "DISCLAIMER: lo scarto sopra e' atteso/noto per il catalogo "
        "placeholder (vedi nota critic-fisico in collasso/catalog.py: i 4 "
        "parametri rho_c, R, n, massa_nucleo non sono mutuamente "
        "consistenti come singolo profilo di Lane-Emden per il catalogo "
        "placeholder). Non e' un difetto del solver di Lane-Emden: qui la "
        "coppia primaria usata per fissare la scala e' sempre "
        "(densita_centrale_gcm3, massa_nucleo_msun); raggio_iniziale_km "
        "resta solo un cross-check indipendente, mai un vincolo simultaneo."
    )


def main() -> None:
    print("=== Validazione Lane-Emden contro soluzioni analitiche ===")
    _validate_against_analytic(n=0, xi1_an=xi1_analytic(0))
    _validate_against_analytic(n=1, xi1_an=xi1_analytic(1))
    _validate_n5()

    print()
    print("=== Profilo fisico stella scelta (s20) ===")
    _physical_profile_stella_scelta()


if __name__ == "__main__":
    main()
