"""
tests/test_eos.py — test dell'EOS di Chandrasekhar (Step 3).

Tolleranze fissate dal planner in STATUS.md, "Piano ciclo corrente (Step 3)
— REV. 2", sezione 3.1 (non a discrezione del coder).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collasso.eos import (
    X_SERIES_THRESHOLD,
    _gamma1_of_x,
    _rho_from_x,
    chandrasekhar_f,
    chandrasekhar_k_deviation_pct,
    chandrasekhar_mass_msun,
    fermi_x,
    gamma1_chandrasekhar,
    k_ultrarelativistic,
    n_eff_chandrasekhar,
    pressure_chandrasekhar,
    pressure_non_relativistic,
    pressure_ultrarelativistic,
)
from collasso.lane_emden import physical_profile, solve_lane_emden

# --- Tolleranze fissate dal planner (STATUS.md, Step 3, punto 3.1) --------

# Usata per i 4 controlli asintotici pure-x (chandrasekhar_f e _gamma1_of_x
# a x=1e-4 e x=1e4). A x=1e-4 lo scarto atteso e' O(x^2)~1e-8; a x=1e4 lo
# scarto atteso e' O(1/x^2)~1e-8; 1e-6 lascia due ordini di grandezza di
# margine. Affidabile a x=1e-4 SOLO grazie al ramo a serie di
# chandrasekhar_f (senza il fix, cancellazione catastrofica, errore >100%).
TOL_ASYMPT_RTOL = 1e-6

# Tolleranza relativa per il test di continuita' del branching serie/diretta
# di chandrasekhar_f attorno a X_SERIES_THRESHOLD (nuovo in REV. 2).
TOL_BRANCH_CONTINUITY_RTOL = 1e-6

# chandrasekhar_mass_msun(0.5) atteso in questo range (valore atteso dalla
# formula ~1.458 Msun; cross-check con la relazione approssimata di
# letteratura M_Ch ~ 5.83*Ye^2 Msun, Shapiro & Teukolsky 1983, cap. 3, che
# per Ye=0.5 da' ~1.4575 Msun).
TOL_MCH_MIN_MSUN = 1.3
TOL_MCH_MAX_MSUN = 1.6

# Cross-check a differenze finite centrali in log-log dell'indice
# adiabatico Gamma1, valutato in piena regione di transizione (non in un
# limite asintotico).
H_FD_REL = 1e-4
TOL_GAMMA1_FD_RTOL = 1e-5


def test_chandrasekhar_f_asymptotics():
    """f(x)/x^5 -> 8/5 per x->0; f(x)/x^4 -> 2 per x->inf."""
    x_small = 1e-4
    assert abs(chandrasekhar_f(x_small) / x_small**5 - 8.0 / 5.0) / (8.0 / 5.0) < TOL_ASYMPT_RTOL

    x_large = 1e4
    assert abs(chandrasekhar_f(x_large) / x_large**4 - 2.0) / 2.0 < TOL_ASYMPT_RTOL


def test_gamma1_of_x_asymptotics():
    """Gamma1(x) -> 5/3 per x->0 (NR); Gamma1(x) -> 4/3 per x->inf (UR).

    Testato "in forma pura su x" tramite l'helper privato _gamma1_of_x,
    come richiesto esplicitamente dal piano.
    """
    x_small = 1e-4
    assert abs(_gamma1_of_x(x_small) - 5.0 / 3.0) / (5.0 / 3.0) < TOL_ASYMPT_RTOL

    x_large = 1e4
    assert abs(_gamma1_of_x(x_large) - 4.0 / 3.0) / (4.0 / 3.0) < TOL_ASYMPT_RTOL


def test_pressure_chandrasekhar_monotonic_in_rho():
    """P_esatta(rho) e' strettamente crescente in rho a ye fisso.

    Nessuna slack numerica ammessa: a differenza di Lane-Emden non c'e'
    integrazione ODE vicino a una singolarita', la funzione e' analitica
    liscia in rho a ye fisso.
    """
    rho_grid = np.logspace(2, 12, 50)
    p = pressure_chandrasekhar(rho_grid, 0.5)
    assert np.all(np.diff(p) > 0)


def test_chandrasekhar_mass_msun_ye_half():
    m_ch = chandrasekhar_mass_msun(0.5)
    assert TOL_MCH_MIN_MSUN <= m_ch <= TOL_MCH_MAX_MSUN


@pytest.mark.parametrize("rho", [1e6, 1e7])
def test_gamma1_finite_difference_crosscheck(rho):
    """Cross-check a differenze finite centrali in log-log di Gamma1,
    valutato a rho in {1e6, 1e7} g/cm^3 (ye=0.5), corrispondenti a x~0.80
    e x~1.72 — regione di transizione, non asintotica.
    """
    ye = 0.5
    rho_plus = rho * (1.0 + H_FD_REL)
    rho_minus = rho * (1.0 - H_FD_REL)

    p_plus = pressure_chandrasekhar(rho_plus, ye)
    p_minus = pressure_chandrasekhar(rho_minus, ye)

    gamma1_fd = (np.log(p_plus) - np.log(p_minus)) / np.log(rho_plus / rho_minus)
    gamma1_analitico = gamma1_chandrasekhar(rho, ye)

    assert abs(gamma1_fd - gamma1_analitico) / gamma1_analitico < TOL_GAMMA1_FD_RTOL


def test_invalid_rho_raises():
    """rho<=0 deve sollevare ValueError per tutte le funzioni che validano rho."""
    for rho_invalido in (0.0, -1.0):
        with pytest.raises(ValueError):
            fermi_x(rho_invalido, 0.5)
        with pytest.raises(ValueError):
            pressure_chandrasekhar(rho_invalido, 0.5)
        with pytest.raises(ValueError):
            gamma1_chandrasekhar(rho_invalido, 0.5)


def test_invalid_ye_raises():
    """ye<=0 o ye>1 deve sollevare ValueError per le funzioni che validano ye."""
    for ye_invalido in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            fermi_x(1e6, ye_invalido)
        with pytest.raises(ValueError):
            pressure_chandrasekhar(1e6, ye_invalido)
        with pytest.raises(ValueError):
            k_ultrarelativistic(ye_invalido)
        with pytest.raises(ValueError):
            chandrasekhar_mass_msun(ye_invalido)


def test_pressure_matches_nr_and_ur_limits():
    """Cross-check dei PREFATTORI: P_esatta deve combaciare con P_NR a
    x=1e-4 e con P_UR a x=1e4 (non solo la forma adimensionale di
    chandrasekhar_f, ma anche i prefattori numerici delle tre funzioni
    esposte). Affidabile a x=1e-4 grazie al ramo a serie di
    chandrasekhar_f (REV. 2).
    """
    ye = 0.5

    rho_small = _rho_from_x(1e-4, ye)
    p_esatta_small = pressure_chandrasekhar(rho_small, ye)
    p_nr = pressure_non_relativistic(rho_small, ye)
    assert abs(p_esatta_small - p_nr) / p_esatta_small < TOL_ASYMPT_RTOL

    rho_large = _rho_from_x(1e4, ye)
    p_esatta_large = pressure_chandrasekhar(rho_large, ye)
    p_ur = pressure_ultrarelativistic(rho_large, ye)
    assert abs(p_esatta_large - p_ur) / p_esatta_large < TOL_ASYMPT_RTOL


def test_chandrasekhar_f_series_branch_continuity():
    """Verifica dedicata del branching di chandrasekhar_f attorno a
    X_SERIES_THRESHOLD=0.1 (REV. 2, fix cancellazione catastrofica).

    Le formule di riferimento (diretta e a serie) sono scritte
    esplicitamente qui, NON importate come helper privati del modulo, in
    modo che il test sia un controllo indipendente e non un semplice
    richiamo circolare alla stessa implementazione.
    """
    assert X_SERIES_THRESHOLD == 0.1

    # x=0.099, appena SOTTO la soglia: la produzione usa il ramo a serie.
    # A x~0.1 la formula diretta non soffre ancora di cancellazione
    # catastrofica rilevante (errore atteso ~1e-11 relativo), quindi e'
    # un riferimento indipendente valido per il confronto.
    x_sotto = 0.099
    f_diretta_test = x_sotto * (2.0 * x_sotto**2 - 3.0) * np.sqrt(1.0 + x_sotto**2) + 3.0 * np.arcsinh(x_sotto)
    assert abs(chandrasekhar_f(x_sotto) - f_diretta_test) / f_diretta_test < TOL_BRANCH_CONTINUITY_RTOL

    # x=0.101, appena SOPRA la soglia: la produzione usa il ramo diretto.
    x_sopra = 0.101
    f_serie_test = (8.0 / 5.0) * x_sopra**5 - (4.0 / 7.0) * x_sopra**7 + (1.0 / 3.0) * x_sopra**9
    assert abs(chandrasekhar_f(x_sopra) - f_serie_test) / chandrasekhar_f(x_sopra) < TOL_BRANCH_CONTINUITY_RTOL


def test_n_eff_chandrasekhar_asymptotics():
    """n_eff -> 3/2 per x piccolo, n_eff -> 3 per x grande (transizione
    EOS non-relativistica -> ultra-relativistica, controllo di coerenza
    diretto sulla funzione pubblica esposta al catalogo/demo).
    """
    ye = 0.5
    rho_small = _rho_from_x(1e-4, ye)
    rho_large = _rho_from_x(1e4, ye)

    n_eff_small = n_eff_chandrasekhar(rho_small, ye)
    n_eff_large = n_eff_chandrasekhar(rho_large, ye)

    assert abs(n_eff_small - 1.5) / 1.5 < 1e-4
    assert abs(n_eff_large - 3.0) / 3.0 < 1e-4


# --- chandrasekhar_k_deviation_pct (ciclo "Integrazione catalogo reale",
#     successivo a Step 8) -------------------------------------------------


def test_k_deviation_zero_for_genuine_eos_consistent_profile():
    """Scarto atteso ~0 quando alpha_cm e' costruito USANDO la costante K
    reale della EOS (non la scorciatoia di physical_profile che fissa
    alpha per far tornare una massa target arbitraria) — controllo di
    coerenza diretto sulla formula, indipendente dal catalogo.
    """
    ye = 0.495
    rho_c = 7.5e9
    n = n_eff_chandrasekhar(rho_c, ye)
    sol = solve_lane_emden(n)

    k_eos = pressure_chandrasekhar(rho_c, ye) / rho_c ** (1.0 + 1.0 / n)
    from collasso.constants import G_CGS

    alpha_cm_eos_consistent = ((n + 1.0) * k_eos * rho_c ** (1.0 / n - 1.0) / (4.0 * np.pi * G_CGS)) ** 0.5

    scarto = chandrasekhar_k_deviation_pct(rho_c, ye, n, alpha_cm_eos_consistent)
    assert abs(scarto) < 1e-8


def test_k_deviation_positive_for_supra_chandrasekhar_mass():
    """Per una massa target SOPRA M_Ch (nessun vero equilibrio esiste, vedi
    docstring di chandrasekhar_k_deviation_pct), il profilo scalato per
    massa (physical_profile) implica una K piu' 'rigida' della EOS reale:
    scarto atteso strettamente positivo.
    """
    ye = 0.495
    rho_c = 7.5e9
    massa_sopra_mch = 1.46  # s20, ~102.3% di M_Ch(0.495)=1.4270
    n = n_eff_chandrasekhar(rho_c, ye)
    sol = solve_lane_emden(n)
    profile = physical_profile(sol, rho_c, massa_sopra_mch)

    scarto = chandrasekhar_k_deviation_pct(rho_c, ye, n, profile.alpha_cm)
    assert scarto > 0.0
    assert abs(scarto - 2.3) < 0.5  # ordine di grandezza atteso, vedi STATUS.md
