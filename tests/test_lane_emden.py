"""
tests/test_lane_emden.py — test del solver di Lane-Emden (Step 2).

Tolleranze fissate dal planner (STATUS.md, "Piano ciclo corrente (Step 2)",
punto 3.1) — NON a discrezione del coder.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from collasso.lane_emden import physical_profile, solve_lane_emden, theta_analytic, xi1_analytic
from collasso.constants import M_SUN_G

# --- Tolleranze di test (punto 3.1 del piano Step 2) ------------------------
TOL_THETA_RTOL = 1e-6
TOL_THETA_ATOL = 1e-8
TOL_XI1_ABS = 1e-5
TOL_THETA_N5_RTOL = 1e-5
TOL_THETA_N5_ATOL = 1e-7
TOL_THETA0_ABS = 1e-6
TOL_MONOTONE = 1e-10
XI1_SANITY_MAX = 10.0
# Soglia inferiore ammessa per solution.theta.min(): l'array theta pubblico
# NON ha il clipping applicato (vedi nota in LaneEmdenSolution.theta),
# quindi puo' presentare un residuo numerico leggermente negativo (~1e-10)
# vicino a xi1. Il test verifica theta.min() > TOL_THETA_MIN_ABS, non
# theta.min() >= 0.
TOL_THETA_MIN_ABS = -1e-8
# Tolleranza relativa larga per il confronto quantitativo di xi1 numerico
# contro i valori tabulati di letteratura (Chandrasekhar 1939 / Horedt 2004).
XI1_LITERATURE_RTOL = 1e-3
TOL_ALPHA_RTOL = 1e-6


def test_n0_matches_analytic():
    solution = solve_lane_emden(0)
    theta_an = theta_analytic(solution.xi, 0)

    assert np.allclose(solution.theta, theta_an, rtol=TOL_THETA_RTOL, atol=TOL_THETA_ATOL)
    assert abs(solution.xi1 - xi1_analytic(0)) < TOL_XI1_ABS


def test_n1_matches_analytic():
    solution = solve_lane_emden(1)
    theta_an = theta_analytic(solution.xi, 1)

    assert np.allclose(solution.theta, theta_an, rtol=TOL_THETA_RTOL, atol=TOL_THETA_ATOL)
    assert abs(solution.xi1 - xi1_analytic(1)) < TOL_XI1_ABS


def test_n5_no_finite_surface_matches_analytic():
    solution = solve_lane_emden(5, xi_max=50.0)

    assert solution.xi1 is None
    assert solution.dtheta_dxi_at_xi1 is None

    theta_an = (1.0 + solution.xi**2 / 3.0) ** (-0.5)
    assert np.allclose(
        solution.theta, theta_an, rtol=TOL_THETA_N5_RTOL, atol=TOL_THETA_N5_ATOL
    )


@pytest.mark.parametrize("n", [1.5, 2.0, 2.5, 3.0])
def test_solver_properties(n):
    solution = solve_lane_emden(n)

    assert abs(solution.theta[0] - 1.0) < TOL_THETA0_ABS
    assert np.all(np.diff(solution.theta) <= TOL_MONOTONE)

    assert solution.xi1 is not None
    assert solution.xi1 > 0
    assert solution.xi1 < XI1_SANITY_MAX

    # theta pubblico NON ha il clipping applicato (vedi nota nel docstring
    # del campo LaneEmdenSolution.theta): puo' presentare un residuo
    # numerico leggermente negativo (~1e-10) all'ultimo punto di griglia
    # (xi=xi1), quindi qui si verifica theta.min() > TOL_THETA_MIN_ABS
    # (una piccola slack negativa), non theta.min() >= 0 — un controllo
    # theta.min() >= 0 fallirebbe spuriamente per rumore numerico atteso,
    # non per un vero errore del solver.
    assert solution.theta.min() > TOL_THETA_MIN_ABS


# Valori tabulati di letteratura per xi1 (Chandrasekhar, An Introduction to
# the Study of Stellar Structure, 1939; riportati anche in Horedt,
# Polytropes: Applications in Astrophysics and Related Fields, 2004).
_XI1_LITERATURE = {
    1.5: 3.65375,
    2.0: 4.35287,
    2.5: 5.35528,
    3.0: 6.89685,
}


@pytest.mark.parametrize("n", [1.5, 2.0, 2.5, 3.0])
def test_xi1_literature_values(n):
    """Validazione quantitativa di xi1 numerico contro valori tabulati di
    letteratura (Chandrasekhar 1939 / Horedt 2004), complementare (non
    sostitutiva) al controllo di plausibilita' grossolano XI1_SANITY_MAX di
    test_solver_properties.
    """
    solution = solve_lane_emden(n)
    xi1_letteratura = _XI1_LITERATURE[n]

    assert abs(solution.xi1 - xi1_letteratura) / xi1_letteratura < XI1_LITERATURE_RTOL


def test_physical_profile_self_consistent():
    """Controllo di correttezza ALGEBRICA dell'inversione in
    physical_profile (radice cubica, segni, fattori).

    NOTA IMPORTANTE: M_theoretical e' calcolato con la STESSA formula usata
    poi in produzione dentro physical_profile, quindi questo test NON
    costituisce una validazione fisica indipendente della formula stessa
    (non verifica se la formula sia fisicamente corretta, solo se
    l'implementazione la inverte correttamente). Da non scambiare in futuro
    per una verifica piu' forte di quanto sia.
    """
    n_test = 1.5
    solution = solve_lane_emden(n_test)

    alpha_test_cm = 1.0e8
    rho_c_test = 1.0e10

    M_theoretical_g = (
        4.0
        * math.pi
        * alpha_test_cm**3
        * rho_c_test
        * (-(solution.xi1**2) * solution.dtheta_dxi_at_xi1)
    )
    M_theoretical_msun = M_theoretical_g / M_SUN_G

    profile = physical_profile(solution, rho_c_test, M_theoretical_msun)

    assert abs(profile.alpha_cm - alpha_test_cm) / alpha_test_cm < TOL_ALPHA_RTOL


def test_physical_profile_raises_without_finite_surface():
    solution = solve_lane_emden(5, xi_max=50.0)

    with pytest.raises(ValueError):
        physical_profile(solution, rho_c_gcm3=1.0e10, massa_msun=1.5)


def test_negative_n_raises():
    with pytest.raises(ValueError):
        solve_lane_emden(-1)
