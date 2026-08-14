"""
tests/test_relativistic.py — test delle correzioni relativistiche sul
nucleo (Step 5).

Costanti/tolleranze fissate dal planner in STATUS.md, "Piano ciclo corrente
(Step 5)", sezione 5 (non a discrezione del coder), salvo dove
esplicitamente annotato sotto.

NOTA SUL TEST 5.3 (deviazione dal testo letterale del piano, segnalata
esplicitamente, non "aggiustata" silenziosamente): il piano (STATUS.md,
punto 5.3) chiede di verificare che `relativistic_grav_accel` "sollevi
ValueError" quando la compattezza della shell piu' interna e'
>= COMPACTNESS_SAFETY_LIMIT. Questo e' in CONTRADDIZIONE diretta con il
punto 1.2 dello stesso piano (correzione da review, ribadita come vincolo
CRITICO dall'orchestratore): "relativistic_grav_accel NON deve MAI
sollevare un'eccezione", perche' la funzione e' sul path del RHS passato a
solve_ivp (un'eccezione li' non verrebbe catturata da scipy — crash non
gestito). Data l'esplicita priorita' assegnata al vincolo di non sollevare
mai eccezioni (che e' anche l'unico dei due compatibile con l'uso reale
della funzione dentro _shell_dvdt), il test sotto verifica il comportamento
CORRETTO secondo 1.2 (nessuna eccezione, valore finito via clamp
COMPACTNESS_CLAMP_MAX) invece del testo letterale di 5.3. Questa
discrepanza tra 1.2 e 5.3 va segnalata a planner/reviewer.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import collasso.dynamics as dynamics
from collasso.catalog import get_progenitor_by_id, load_reference_catalog
from collasso.constants import C_LIGHT_CGS, G_CGS
from collasso.dynamics import (
    T_MAX_FREE_FALL_MULTIPLIER,
    build_initial_shells,
    free_fall_time_s,
    simulate_collapse,
)
from collasso.lane_emden import physical_profile, solve_lane_emden
from collasso.relativistic import (
    COMPACTNESS_SAFETY_LIMIT,
    relativistic_grav_accel,
)

# --- Costanti di tolleranza (STATUS.md, piano Step 5, sezione 3) -----------

# TOL_WEAK_FIELD_RTOL: CORREZIONE (da review) rispetto alla REV. 1 del
# piano (che proponeva 1e-8): la deviazione a_GR/a_Newt - 1 e' LINEARE in
# compattezza e in P/(rho*c^2) (nessuna cancellazione quadratica), quindi a
# compattezza~1e-8 lo scarto atteso e' ~1e-8 (verificato numericamente dal
# coder: 3.0e-8, vedi test sotto), non ~1e-16. Fissata con margine ampio
# (~300x) per assorbire differenze di coefficiente fra i tre termini
# additivi.
TOL_WEAK_FIELD_RTOL = 1e-5

N_SHELLS_TEST_DEFAULT = 50


def _s20_profile():
    catalogo = load_reference_catalog()
    stella = get_progenitor_by_id(catalogo, "s20")
    solution = solve_lane_emden(n=stella.n_politropico)
    profile = physical_profile(solution, stella.densita_centrale_gcm3, stella.massa_nucleo_msun)
    return stella, profile


# --- 5.1: limite di campo debole --------------------------------------------


def test_weak_field_limit_matches_newtonian():
    """Compattezza E P/(rho*c^2) entrambi ~1e-8 simultaneamente (scelta di
    r_cm/m_g/p_cgs/rho_gcm3 verificata dal coder prima di fissare il test,
    vedi STATUS.md piano Step 5, punto 3.2): scarto osservato ~3.0e-8,
    ampiamente entro TOL_WEAK_FIELD_RTOL=1e-5.
    """
    target = 1.0e-8
    c2 = C_LIGHT_CGS**2

    r_cm = 1.0e9
    m_g = target * r_cm * c2 / (2.0 * G_CGS)
    # rho scelta in modo che anche il termine 4*pi*r^3*P/(m*c^2) sia ~target
    # (altrimenti, con r/m fissati solo dalla compattezza, questo termine
    # puo' dominare per r grande — verificato numericamente dal coder).
    rho_gcm3 = (target * c2 / (8.0 * math.pi * G_CGS)) / r_cm**2
    p_cgs = target * rho_gcm3 * c2

    a_rel = relativistic_grav_accel(m_g, r_cm, p_cgs, rho_gcm3)
    a_newt = -G_CGS * m_g / r_cm**2

    assert abs(a_rel / a_newt - 1.0) < TOL_WEAK_FIELD_RTOL


# --- 5.2: monotonia/segno ----------------------------------------------------


def test_correction_factor_always_at_least_one():
    """Per una griglia di compattezze in (0, 0.8) (pressione nulla, per
    isolare l'effetto della sola compattezza), il fattore di correzione
    |relativistic_grav_accel| / (G_CGS*m_g/r_cm**2) deve essere sempre >= 1
    (con margine 1e-10 per rumore macchina vicino a compattezza->0).
    """
    r_cm = 1.0e8
    c2 = C_LIGHT_CGS**2

    for compattezza_target in np.linspace(0.01, 0.8, 20):
        m_g = compattezza_target * r_cm * c2 / (2.0 * G_CGS)
        a_rel = relativistic_grav_accel(m_g, r_cm, 0.0, 1.0)
        a_newt_mag = G_CGS * m_g / r_cm**2
        fattore = abs(a_rel) / a_newt_mag
        assert fattore >= 1.0 - 1e-10, f"fattore={fattore} < 1 a compattezza={compattezza_target}"


# --- 5.3: guardia/evento -----------------------------------------------------


def test_schwarzschild_proximity_event_and_no_exception_in_accel():
    """Costruisce uno stato con compattezza (di tutte le shell, uguale per
    costruzione) >= COMPACTNESS_SAFETY_LIMIT: verifica che
    _schwarzschild_proximity_event ritorni un valore <= 0 (evento
    scattato).

    Vedi nota di modulo per la deviazione rispetto al testo letterale di
    5.3 sulla seconda parte (relativistic_grav_accel NON deve sollevare
    ValueError, per il vincolo critico 1.2 del piano): qui si verifica
    invece che la funzione ritorni un valore FINITO (clampato), mai
    un'eccezione, come richiesto da 1.2.
    """
    n = 3
    c2 = C_LIGHT_CGS**2
    # 1.5 (non 0.95): >= COMPACTNESS_SAFETY_LIMIT=0.9 (fa scattare l'evento)
    # E >= COMPACTNESS_CLAMP_MAX=0.99 (esercita realmente il clamp dentro
    # relativistic_grav_accel, non solo un valore ancora fisicamente valido
    # come 0.95 sarebbe stato) — coerente col fatto che durante l'evoluzione
    # numerica la compattezza puo' superare 1 prima che solve_ivp elabori
    # l'evento terminale (vedi nota di modulo su solve_ivp).
    compattezza_stato = 1.5

    r_cm = np.array([1.0e6, 2.0e6, 3.0e6])
    m_enclosed_g = compattezza_stato * r_cm * c2 / (2.0 * G_CGS)
    delta_m_g_dummy = np.diff(np.concatenate(([0.0], m_enclosed_g)))
    v = np.zeros(n)
    y = np.concatenate([r_cm, v])

    extra_args = (m_enclosed_g, delta_m_g_dummy, 0.5, n, 1.0e5, True)
    valore_evento = dynamics._schwarzschild_proximity_event(0.0, y, *extra_args)
    assert valore_evento <= 0.0

    # relativistic_grav_accel NON deve sollevare eccezione in questo stato
    # (vincolo critico 1.2): deve ritornare un valore finito, usando il
    # clamp COMPACTNESS_CLAMP_MAX internamente.
    rho_dummy = 1.0e10
    p_dummy = 1.0e30
    a_rel = relativistic_grav_accel(m_enclosed_g[0], r_cm[0], p_dummy, rho_dummy)
    assert np.isfinite(a_rel)


# --- 5.4: regressione (test piu' importante di questo step) -----------------

# FASE A (STATUS.md, piano Step 5, punto 5.4): baseline originaria catturata
# dal coder PRIMA di modificare collasso/dynamics.py, eseguendo
# simulate_collapse con gli stessi parametri di
# test_super_chandrasekhar_triggers_collapse (tests/test_dynamics.py) sul
# codice di Step 4 ancora invariato. Valori a piena precisione (repr di
# Python), NON ricalcolati dopo l'estensione del codice stesso — se lo
# fossero, il test sarebbe tautologico (confronto contro se stesso),
# esattamente cio' che questa procedura in due fasi deve evitare.
#
# RI-BASELINE nel ciclo "Integrazione catalogo reale" (successivo a
# Step 8): questo test confronta relativistic=False contro un riferimento
# fisso per il caso s20 del catalogo — ma s20 ora usa la massa REALE del
# nucleo (O'Connor & Ott 2011) invece del placeholder, e
# T_MAX_FREE_FALL_MULTIPLIER e' stato alzato da 10 a 15 (vedi
# collasso.dynamics), quindi la vecchia baseline (calcolata su dati di
# input diversi) non e' piu' un riferimento valido — il confronto era
# contro dati obsoleti, non un confronto codice-vecchio/codice-nuovo a
# parita' di fisica. Valori qui sotto RI-CATTURATI (non ricalcolati dopo
# una modifica a collasso/dynamics.py stesso, che resta invariato in
# questo ciclo) con relativistic=False sul catalogo aggiornato — vedi
# `data/progenitors_reference.csv` per i valori di ingresso.
BASELINE_T_COLLAPSE_S = 0.3173750339122022
BASELINE_COLLAPSE_REASON = "r_min_threshold"
BASELINE_RHO_C_GCM3_FINAL = 6512070035852.868

TOL_REGRESSION_RTOL = 1e-12


def test_regression_relativistic_false_matches_step4_baseline():
    stella, profile = _s20_profile()
    delta_m_g, _, r0_cm = build_initial_shells(profile.r_cm, profile.rho_gcm3, N_SHELLS_TEST_DEFAULT)

    t_max_s = T_MAX_FREE_FALL_MULTIPLIER * free_fall_time_s(stella.densita_centrale_gcm3)
    result = simulate_collapse(delta_m_g, r0_cm, ye=stella.ye, t_max_s=t_max_s, relativistic=False)

    assert result.collapse_reason == BASELINE_COLLAPSE_REASON
    assert abs(result.t_collapse_s - BASELINE_T_COLLAPSE_S) / BASELINE_T_COLLAPSE_S < TOL_REGRESSION_RTOL
    assert (
        abs(result.rho_c_gcm3_t[-1] - BASELINE_RHO_C_GCM3_FINAL) / BASELINE_RHO_C_GCM3_FINAL
        < TOL_REGRESSION_RTOL
    )


# --- 5.5: confronto fisico atteso -------------------------------------------


def test_relativistic_collapse_is_faster():
    """simulate_collapse(..., relativistic=True) per lo stesso caso s20
    deve avere t_collapse_s MINORE (collasso piu' rapido) di
    relativistic=False — test qualitativo robusto, solo '<', coerente con
    l'effetto fisico atteso (gravita' effettiva piu' forte).
    """
    stella, profile = _s20_profile()
    delta_m_g, _, r0_cm = build_initial_shells(profile.r_cm, profile.rho_gcm3, N_SHELLS_TEST_DEFAULT)

    t_max_s = T_MAX_FREE_FALL_MULTIPLIER * free_fall_time_s(stella.densita_centrale_gcm3)
    result_newt = simulate_collapse(delta_m_g, r0_cm, ye=stella.ye, t_max_s=t_max_s, relativistic=False)
    result_gr = simulate_collapse(delta_m_g, r0_cm, ye=stella.ye, t_max_s=t_max_s, relativistic=True)

    assert result_newt.collapsed is True
    assert result_gr.collapsed is True
    assert result_gr.t_collapse_s < result_newt.t_collapse_s
