"""
tests/test_tov.py — test del limite TOV e della classificazione remnant
(Step 6).

Costanti/tolleranze fissate dal planner in STATUS.md, "Piano ciclo corrente
(Step 6)" (non a discrezione del coder), salvo dove esplicitamente annotato.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import collasso.tov as tov
from collasso.constants import C_LIGHT_CGS, G_CGS
from collasso.eos import chandrasekhar_f
from collasso.eos_neutron import P0_N_CGS, pressure_degenerate_neutron, rho_of_x_neutron
from collasso.eos_piecewise_polytrope import build_apr4, build_sly
from collasso.remnant import classify_remnant
from collasso.tov import (
    COMPACTNESS_TOV_CLAMP_MAX,
    M_TOV_SANITY_RANGE_MSUN,
    RHO_C_GRID_REALISTIC_CGS,
    find_ov_mass_limit,
    find_ov_mass_limit_legacy_bug_rho_source,
    solve_tov,
)

# Stessa tolleranza di monotonia di Lane-Emden (Step 2), piccola slack
# positiva per rumore numerico su una sequenza attesa non-crescente.
TOL_MONOTONE = 1e-10

# --- Retrofit Step 6 — EOS realistica: costanti/tolleranze di test fissate
#     dal planner in STATUS.md, sezione "Retrofit Step 6" -------------------

# Valore storico di Oppenheimer & Volkoff (Phys. Rev. 55, 374, 1939), gas di
# neutroni liberi non interagenti — riferimento del NUOVO test dedicato sul
# path CORRETTO (epsilon/c^2), non piu' un test di regressione bit-identico
# a Step 6 (quel concetto non si applica piu': il bug fix cambia
# intenzionalmente il numero da 0.8429 a ~0.71 Msun, un miglioramento
# documentato, non una regressione da evitare — vedi STATUS.md).
M_OV_HISTORICAL_MSUN = 0.7
TOL_M_OV_CORRECTED_RTOL = 0.10

# Valore del path storico/bacato di Step 6 (rho come sorgente), usato SOLO
# per verificare che il path legacy sia ancora riproducibile identicamente
# (nessuna regressione accidentale sul codice storico preservato per
# confronto) — NON e' piu' il valore "giusto" del progetto.
M_MAX_LEGACY_BUG_MSUN = 0.8429
TOL_M_MAX_LEGACY_RTOL = 0.01

M_MAX_SLY_LITERATURE_MSUN = 2.05
M_MAX_APR4_LITERATURE_MSUN = 2.20
TOL_MMAX_LITERATURE_RTOL = 0.10


# --- 1. Limite di massa di Oppenheimer-Volkoff ------------------------------


def test_m_max_within_historical_sanity_range():
    """M_max_msun deve cadere nel range di plausibilita' fissato dal
    planner (M_TOV_SANITY_RANGE_MSUN = (0.4, 1.2) Msun), tolleranza
    deliberatamente larga rispetto al valore storico O&V 1939 (~0.7 Msun)
    — vedi nota nel piano Step 6, punto 1.10.
    """
    rho_c_max_gcm3, m_max_msun, sequenza = find_ov_mass_limit()

    assert M_TOV_SANITY_RANGE_MSUN[0] < m_max_msun < M_TOV_SANITY_RANGE_MSUN[1]
    assert rho_c_max_gcm3 > 0
    assert len(sequenza) == len(tov.RHO_C_GRID_GCM3)


# --- 2. Proprieta' del solver (nessuna eccezione, monotonia) ---------------


def test_solve_tov_no_exception_and_monotonic_profile():
    """solve_tov non deve sollevare eccezioni per una densita' centrale
    rappresentativa della griglia; r_cm deve essere strettamente crescente
    (variabile indipendente), x deve essere non-crescente (la pressione
    scende monotonamente dal centro alla superficie).
    """
    soluzione = solve_tov(1.0e15)

    assert np.all(np.diff(soluzione.r_cm) > 0)
    # Piccola slack positiva per rumore numerico (stesso principio di
    # TOL_MONOTONE in Lane-Emden, Step 2).
    assert np.all(np.diff(soluzione.x) <= TOL_MONOTONE)
    assert soluzione.R_cm > 0
    assert soluzione.M_g > 0
    assert soluzione.M_msun > 0


def test_solve_tov_raises_on_non_positive_rho_c():
    with pytest.raises(ValueError):
        solve_tov(0.0)
    with pytest.raises(ValueError):
        solve_tov(-1.0)


# --- 3. RHS TOV: mai eccezioni, anche per stati numericamente estremi ------
#     (STATUS.md, piano Step 6, punto 1.3ter — CRITICO, verificato dal
#     coder come richiesto esplicitamente dall'orchestratore).


def test_tov_rhs_never_raises_for_negative_x_and_high_compactness():
    """Chiama _tov_rhs direttamente con x negativo e con una compattezza
    2*G*m/(r*c^2) >= COMPACTNESS_TOV_CLAMP_MAX: deve ritornare una lista di
    2 valori finiti, senza sollevare alcuna eccezione (entrambi i clamp
    incondizionati del RHS devono attivarsi correttamente).
    """
    r_cm = 1.0e6
    c2 = C_LIGHT_CGS**2

    # m_g scelto in modo che la compattezza raw sia 1.5x il clamp massimo
    # (esercita realmente il clamp, non solo un valore ancora sub-soglia).
    m_g = 1.5 * COMPACTNESS_TOV_CLAMP_MAX * r_cm * c2 / (2.0 * G_CGS)
    x_negativo = -0.5

    risultato = tov._tov_rhs(r_cm, [m_g, x_negativo])

    assert len(risultato) == 2
    assert all(np.isfinite(v) for v in risultato)


def test_tov_rhs_piecewise_never_raises_for_negative_rho_and_high_compactness():
    """Stesso test del punto 3 sopra, ma per il RHS del path piecewise
    polytrope (Retrofit Step 6, `_tov_rhs_piecewise`) — segnalato dal
    critic-fisico come lacuna di copertura: il clamp incondizionato su
    rho e quello sulla compattezza erano presenti nel codice ma non
    esercitati insieme da un test dedicato, a differenza del path a gas
    libero sopra. Verifica che entrambi i clamp si attivino correttamente
    senza mai sollevare eccezioni (stesso principio critico di Step 4/5/6:
    il RHS passato a solve_ivp non deve mai propagare un'eccezione).
    """
    r_cm = 1.0e6
    c2 = C_LIGHT_CGS**2

    m_g = 1.5 * COMPACTNESS_TOV_CLAMP_MAX * r_cm * c2 / (2.0 * G_CGS)
    rho_negativa = -1.0e10

    eos = build_sly()
    risultato = tov._tov_rhs_piecewise(r_cm, [m_g, rho_negativa], eos)

    assert len(risultato) == 2
    assert all(np.isfinite(v) for v in risultato)


# --- 4. classify_remnant (logica pura) --------------------------------------


def test_classify_remnant_three_cases_and_boundaries():
    m_ch = 1.4
    m_tov = 2.0

    # Sotto Chandrasekhar.
    assert classify_remnant(1.0, m_ch, m_tov) == "white_dwarf"
    # Fra Chandrasekhar e TOV.
    assert classify_remnant(1.8, m_ch, m_tov) == "neutron_star"
    # Sopra TOV.
    assert classify_remnant(2.5, m_ch, m_tov) == "black_hole"

    # Valori di confine (fissati dal piano: "<" per Chandrasekhar, "<" per TOV).
    assert classify_remnant(m_ch, m_ch, m_tov) == "neutron_star"
    assert classify_remnant(m_tov, m_ch, m_tov) == "black_hole"


# --- 5. Riuso genuino di chandrasekhar_f in pressure_degenerate_neutron ----


def test_pressure_degenerate_neutron_reuses_chandrasekhar_f():
    """Verifica che `pressure_degenerate_neutron` richiami DAVVERO
    `collasso.eos.chandrasekhar_f` (import diretto), non una copia
    duplicata del codice: chandrasekhar_f(x) chiamato direttamente deve
    coincidere con pressure_degenerate_neutron(rho_of_x_neutron(x)) /
    P0_N_CGS, per una griglia di x che copre entrambi i rami (serie e
    diretto) di chandrasekhar_f.

    NOTA (deviazione minima dal testo letterale del piano, segnalata
    esplicitamente, non "aggiustata" silenziosamente): il piano (STATUS.md,
    Step 6, sezione 6, ultimo punto) chiede che i due valori "coincidano
    esattamente". In pratica il confronto passa per il round-trip
    x -> rho_of_x_neutron(x) -> fermi_x_neutron(rho) (dentro
    pressure_degenerate_neutron), che e' un'inversione algebrica esatta ma
    NON bit-esatta in floating point (elevamento al cubo seguito da radice
    cubica tramite **(1/3) non e' garantito essere l'inverso esatto in
    virgola mobile) — osservato uno scarto relativo fino a ~1.4e-7 a
    x=0.1 (proprio la soglia X_SERIES_THRESHOLD fra i due rami). Questo NON
    e' un segno di duplicazione della logica (il test verifica comunque che
    entrambi i rami di chandrasekhar_f siano raggiunti tramite la stessa
    funzione riusata): la verifica qui e' quindi ravvicinata
    (np.allclose, rtol=1e-6) invece che a bit esatti.
    """
    x_grid = np.array([1e-4, 0.05, 0.1, 0.5, 1.0, 10.0, 1e4])

    f_diretto = chandrasekhar_f(x_grid)
    rho = rho_of_x_neutron(x_grid)
    f_via_pressure = pressure_degenerate_neutron(rho) / P0_N_CGS

    assert np.allclose(f_diretto, f_via_pressure, rtol=1e-6, atol=0.0)


# --- 6. Retrofit Step 6 — correzione del bug fisico (rho -> epsilon/c^2) ---


def test_m_max_free_neutron_gas_corrected_close_to_historical_value():
    """NUOVO test (Retrofit Step 6), sostituisce il vecchio concetto di
    "regressione bit-identica a Step 6" (non piu' applicabile: il bug fix
    e' un cambiamento INTENZIONALE, documentato — vedi STATUS.md,
    "CORREZIONE POST-REVIEW"). Il path CORRETTO (eos=None, sorgente
    epsilon/c^2 in dP/dr e dm/dr) deve dare M_max entro il 10% dal valore
    storico di Oppenheimer & Volkoff 1939 (~0.7 Msun) — molto piu' stretto
    della vecchia tolleranza di plausibilita' larga (0.4-1.2 Msun,
    M_TOV_SANITY_RANGE_MSUN, che resta comunque valida e testata sopra).
    """
    _, m_max_msun, _ = find_ov_mass_limit()

    assert abs(m_max_msun - M_OV_HISTORICAL_MSUN) / M_OV_HISTORICAL_MSUN < TOL_M_OV_CORRECTED_RTOL


def test_m_max_free_neutron_gas_legacy_bug_reproducible():
    """Verifica che il path storico/bacato (`find_ov_mass_limit_legacy_bug_rho_source`,
    preservato SOLO per il confronto onesto nello script demo) sia ancora
    riproducibile al valore noto (0.8429 Msun, entro 1%) — NON e' un
    endorsement del valore (che resta fisicamente scorretto), solo una
    verifica che il codice storico preservato non sia stato alterato per
    errore.
    """
    _, m_max_legacy_msun, _ = find_ov_mass_limit_legacy_bug_rho_source()

    assert abs(m_max_legacy_msun - M_MAX_LEGACY_BUG_MSUN) / M_MAX_LEGACY_BUG_MSUN < TOL_M_MAX_LEGACY_RTOL


def test_corrected_path_gives_lower_mass_than_legacy_bug_path():
    """Verifica esplicita e diretta del segno dell'effetto del bug fix:
    la sorgente fisicamente corretta (epsilon/c^2, gas piu' "pesante"
    gravitazionalmente per un dato rho, essendo epsilon >= rho*c^2) deve
    dare un M_max STRETTAMENTE inferiore al path storico/bacato (rho come
    sorgente) — coerente con 0.7100 < 0.8427 riportato in STATUS.md.
    """
    _, m_max_corrected, _ = find_ov_mass_limit()
    _, m_max_legacy, _ = find_ov_mass_limit_legacy_bug_rho_source()

    assert m_max_corrected < m_max_legacy


# --- 7. Retrofit Step 6 — EOS realistiche (SLy/APR4) ------------------------


def test_m_max_sly_within_tolerance_of_literature():
    """M_max_msun per SLy entro +-10% del valore di letteratura (Read et
    al. 2009 e successivi, ~2.05 Msun)."""
    eos = build_sly()
    _, m_max_msun, sequenza = find_ov_mass_limit(RHO_C_GRID_REALISTIC_CGS, eos=eos)

    assert abs(m_max_msun - M_MAX_SLY_LITERATURE_MSUN) / M_MAX_SLY_LITERATURE_MSUN < TOL_MMAX_LITERATURE_RTOL
    assert len(sequenza) == len(RHO_C_GRID_REALISTIC_CGS)
    # Il profilo di soluzione per una EOS piecewise-polytrope popola
    # rho_gcm3, non x (che resta None per questo path — vedi TOVSolution).
    assert sequenza[0].rho_gcm3 is not None
    assert sequenza[0].x is None


def test_m_max_apr4_within_tolerance_of_literature():
    """M_max_msun per APR4 entro +-10% del valore di letteratura (~2.20
    Msun)."""
    eos = build_apr4()
    _, m_max_msun, _ = find_ov_mass_limit(RHO_C_GRID_REALISTIC_CGS, eos=eos)

    assert abs(m_max_msun - M_MAX_APR4_LITERATURE_MSUN) / M_MAX_APR4_LITERATURE_MSUN < TOL_MMAX_LITERATURE_RTOL


def test_sly_tolerance_sensitivity_m_max_stable():
    """Controllo di sensibilita' alle tolleranze del solver (richiesto dal
    piano, Retrofit Step 6): dimezzare rtol/atol per SLy non deve cambiare
    M_max in modo apprezzabile (le 6 discontinuita' di dP_drho ai confini
    fra pezzi polytropici sono un fatto noto/atteso, C0 ma non C1 — non un
    errore, ma verificato empiricamente qui). Tolleranza di stabilita'
    fissata larga (0.1%) per assorbire rumore numerico residuo, molto piu'
    stretta di qualunque tolleranza di plausibilita' fisica.
    """
    eos = build_sly()
    rho_c_grid = RHO_C_GRID_REALISTIC_CGS

    _, m_max_default, _ = find_ov_mass_limit(rho_c_grid, eos=eos, rtol=1e-8, atol=1e-6)
    _, m_max_half_tol, _ = find_ov_mass_limit(rho_c_grid, eos=eos, rtol=5e-9, atol=5e-7)

    assert abs(m_max_default - m_max_half_tol) / m_max_default < 1e-3
