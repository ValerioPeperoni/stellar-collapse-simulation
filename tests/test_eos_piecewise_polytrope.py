"""
tests/test_eos_piecewise_polytrope.py — test della EOS a politropi a tratti
(Retrofit Step 6, Read et al. 2009, fit SLy/APR4).

Costanti/tolleranze fissate dal planner in STATUS.md, "Retrofit Step 6 —
EOS realistica" (non a discrezione del coder), salvo dove esplicitamente
annotato.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collasso.constants import C_LIGHT_CGS
from collasso.eos_piecewise_polytrope import (
    APR4,
    GAMMA_LOW,
    K_LOW_CGS,
    RHO_1_CGS,
    RHO_2_CGS,
    RHO_LOW_CGS,
    SLY,
    _build_piecewise_polytrope,
    build_apr4,
    build_sly,
    dP_drho,
    energy_density_of_rho,
    pressure_of_rho,
)

# Tolleranza per i confronti di continuita' ai confini fra pezzi: i K_LOW
# della crosta (Read et al. Tabella II) sono citati con precisione finita
# (~10-12 cifre), quindi la continuita' di P e' esatta solo entro l'errore
# di arrotondamento della tabella stessa, non a precisione macchina piena
# (osservato ~1e-7 relativo ai confini di crosta, ~1e-10 o meglio ai
# confini del nucleo, dove K1/K2/K3 sono derivati algebricamente qui).
TOL_CONTINUITY_RTOL = 1e-5

# Tolleranza per il cross-check di dP_drho a differenze finite.
TOL_DPDRHO_FD_RTOL = 1e-6
H_FD_REL = 1e-6

# Tolleranza per il confronto di M_max contro la letteratura (Read et al.
# 2009 e successivi): fissata larga (10%) perche' la parametrizzazione e'
# fenomenologica, non un fit a precisione arbitraria.
TOL_MMAX_LITERATURE_RTOL = 0.10
M_MAX_SLY_LITERATURE_MSUN = 2.05
M_MAX_APR4_LITERATURE_MSUN = 2.20


# --- 1. Costruzione: reproduzione esatta dei parametri di tabella ----------


@pytest.mark.parametrize("build_fn,params", [(build_sly, SLY), (build_apr4, APR4)])
def test_build_reproduces_table_parameters(build_fn, params):
    """Verifica che k1, k2, k3, rho0 siano ricalcolati in modo indipendente
    nel test (stesse formule del piano, riscritte qui senza chiamare
    l'implementazione) e coincidano con quanto prodotto da build_sly()/
    build_apr4() — protegge contro errori di trascrizione delle formule
    nell'implementazione.
    """
    eos = build_fn()

    p1 = 10.0 ** params["log10_p1"]
    k1_atteso = p1 / RHO_1_CGS ** params["gamma1"]
    k2_atteso = p1 / RHO_1_CGS ** params["gamma2"]
    k3_atteso = k2_atteso * RHO_2_CGS ** (params["gamma2"] - params["gamma3"])
    rho0_atteso = (K_LOW_CGS[3] / k1_atteso) ** (1.0 / (params["gamma1"] - GAMMA_LOW[3]))

    assert eos.K[4] == pytest.approx(k1_atteso, rel=1e-12)
    assert eos.K[5] == pytest.approx(k2_atteso, rel=1e-12)
    assert eos.K[6] == pytest.approx(k3_atteso, rel=1e-12)
    assert eos.rho_boundaries[4] == pytest.approx(rho0_atteso, rel=1e-12)

    # rho0 deve cadere strettamente fra RHO_LOW_CGS[3] e RHO_1_CGS (condizione
    # di applicabilita' della costruzione a 7 pezzi).
    assert RHO_LOW_CGS[3] < eos.rho_boundaries[4] < RHO_1_CGS

    # I confini fissi di crosta e nucleo devono coincidere con le costanti
    # di modulo.
    assert eos.rho_boundaries[0] == 0.0
    assert eos.rho_boundaries[1] == RHO_LOW_CGS[1]
    assert eos.rho_boundaries[2] == RHO_LOW_CGS[2]
    assert eos.rho_boundaries[3] == RHO_LOW_CGS[3]
    assert eos.rho_boundaries[5] == RHO_1_CGS
    assert eos.rho_boundaries[6] == RHO_2_CGS


def test_build_raises_when_rho0_out_of_range():
    """Se rho0 non cade fra RHO_LOW_CGS[3] e RHO_1_CGS, il costruttore deve
    sollevare ValueError esplicito (caso a 8 pezzi non implementato) —
    esercitato con un log10_p1 artificialmente troppo alto (rho0 troppo
    piccolo, esce sotto RHO_LOW_CGS[3]).
    """
    with pytest.raises(ValueError):
        _build_piecewise_polytrope("fuori_range", log10_p1=40.0, gamma1=3.0, gamma2=3.0, gamma3=3.0)


# --- 2. Continuita' di P e epsilon ai 7 confini -----------------------------


@pytest.mark.parametrize("build_fn", [build_sly, build_apr4])
def test_pressure_continuous_at_all_boundaries(build_fn):
    """P valutata con le costanti del pezzo (i-1) al confine i deve
    coincidere con P valutata tramite pressure_of_rho (che usa le costanti
    del pezzo i, il pezzo che INIZIA in quel confine), per tutti i 6 confini
    interni (i=1..6).
    """
    eos = build_fn()
    for i in range(1, 7):
        rho_b = eos.rho_boundaries[i]
        p_sotto = eos.K[i - 1] * rho_b ** eos.Gamma[i - 1]
        p_sopra = pressure_of_rho(eos, rho_b)
        assert p_sopra == pytest.approx(p_sotto, rel=TOL_CONTINUITY_RTOL)


@pytest.mark.parametrize("build_fn", [build_sly, build_apr4])
def test_energy_density_continuous_at_all_boundaries(build_fn):
    """epsilon valutata con le costanti del pezzo (i-1) al confine i deve
    coincidere con epsilon valutata tramite energy_density_of_rho (pezzo
    i), per tutti i 6 confini interni.
    """
    eos = build_fn()
    for i in range(1, 7):
        rho_b = eos.rho_boundaries[i]
        p_sotto = eos.K[i - 1] * rho_b ** eos.Gamma[i - 1]
        eps_sotto = (1.0 + eos.a[i - 1]) * rho_b * C_LIGHT_CGS**2 + eos.n[i - 1] * p_sotto
        eps_sopra = energy_density_of_rho(eos, rho_b)
        assert eps_sopra == pytest.approx(eps_sotto, rel=TOL_CONTINUITY_RTOL)


def test_pressure_at_rho_zero_is_zero():
    """P(0)=0 per costruzione (rho_boundaries[0]=0, GAMMA_LOW[0]>0)."""
    eos = build_sly()
    assert eos.P_boundaries[0] == 0.0


# --- 3. dP_drho: forma chiusa vs differenze finite --------------------------


@pytest.mark.parametrize("build_fn", [build_sly, build_apr4])
def test_dp_drho_matches_finite_difference(build_fn):
    """dP_drho (forma chiusa) deve coincidere con una differenza finita
    centrata di pressure_of_rho, su una griglia che copre crosta e nucleo
    (lontano dai confini, dove dP_drho e' discontinua per costruzione — vedi
    docstring di dP_drho).
    """
    eos = build_fn()
    rho_grid = np.array([1e6, 1e9, 1e12, 1e13, 3e14, 8e14, 3e15])
    h = rho_grid * H_FD_REL

    fd = (pressure_of_rho(eos, rho_grid + h) - pressure_of_rho(eos, rho_grid - h)) / (2.0 * h)
    analitico = dP_drho(eos, rho_grid)

    assert np.allclose(fd, analitico, rtol=TOL_DPDRHO_FD_RTOL)


# --- 4. Validazione input ----------------------------------------------------


def test_pressure_of_rho_raises_on_non_positive_rho():
    eos = build_sly()
    with pytest.raises(ValueError):
        pressure_of_rho(eos, 0.0)
    with pytest.raises(ValueError):
        pressure_of_rho(eos, -1.0)


# --- 5. Plausibilita' di M_max contro letteratura (cross-check TOV) --------


def test_m_max_sly_close_to_literature():
    """Cross-check end-to-end: il limite di massa TOV con EOS SLy deve
    cadere entro TOL_MMAX_LITERATURE_RTOL (10%) dal valore di letteratura
    (Read et al. 2009 e successivi, ~2.05 Msun). Richiede collasso.tov,
    import locale per evitare dipendenza circolare a livello di modulo.
    """
    from collasso.tov import RHO_C_GRID_REALISTIC_CGS, find_ov_mass_limit

    eos = build_sly()
    _, m_max_msun, _ = find_ov_mass_limit(RHO_C_GRID_REALISTIC_CGS, eos=eos)

    assert abs(m_max_msun - M_MAX_SLY_LITERATURE_MSUN) / M_MAX_SLY_LITERATURE_MSUN < TOL_MMAX_LITERATURE_RTOL


def test_m_max_apr4_close_to_literature():
    """Come sopra, per APR4 (~2.20 Msun)."""
    from collasso.tov import RHO_C_GRID_REALISTIC_CGS, find_ov_mass_limit

    eos = build_apr4()
    _, m_max_msun, _ = find_ov_mass_limit(RHO_C_GRID_REALISTIC_CGS, eos=eos)

    assert abs(m_max_msun - M_MAX_APR4_LITERATURE_MSUN) / M_MAX_APR4_LITERATURE_MSUN < TOL_MMAX_LITERATURE_RTOL
