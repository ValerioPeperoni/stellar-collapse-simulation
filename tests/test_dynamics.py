"""
tests/test_dynamics.py — test della dinamica shell Lagrangiane del collasso
(Step 4).

Costanti/tolleranze fissate dal planner in STATUS.md, "Piano ciclo corrente
(Step 4) — REV. 2", "REV. 3" e "REV. 4" (fix successivi mirati al test di
convergenza 3.5 e alla scala di massa M_TEST_MSUN/M_SUB_CHANDRA_TEST_MSUN),
sezione 3 (non a discrezione del coder). Le tolleranze GLOBAL_RATIO_TOL_N400
e GLOBAL_RATIO_REDUCTION_FACTOR_MIN sono state fissate empiricamente dal
coder (REV. 4, punto 3 del piano autorizza esplicitamente valori diversi dai
punti di partenza suggeriti se i numeri osservati lo richiedono) — vedi
docstring del test 3.5 per i valori osservati e la motivazione.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import collasso.dynamics as dynamics
from collasso import eos
from collasso.catalog import get_progenitor_by_id, load_reference_catalog
from collasso.constants import G_CGS, M_SUN_G
from collasso.dynamics import (
    T_MAX_FREE_FALL_MULTIPLIER,
    V_FLOOR_CM3,
    build_initial_shells,
    free_fall_time_s,
    polytrope_equilibrium_rho_c_gcm3,
    simulate_collapse,
)
from collasso.lane_emden import physical_profile, solve_lane_emden

# --- Costanti di tolleranza e parametri di test (STATUS.md, Step 4, 3.1) ---

TOL_MASS_SHELLS_INTERNAL_RTOL = 1e-8
TOL_MASS_SHELLS_EXTERNAL_RTOL = 1e-3
TOL_R_LAST_RTOL = 1e-3

# RATIO_TOL_N400 / RATIO_REDUCTION_FACTOR_MIN (REV. 2) — SUPERSEDUTE (REV. 4,
# STATUS.md piano Step 4). Non piu' usate nella logica del test 3.5,
# sostituite da GLOBAL_RATIO_TOL_N400 / GLOBAL_RATIO_REDUCTION_FACTOR_MIN
# sotto. Mantenute solo come nota storica, NON come costanti attive.
# RATIO_TOL_N400 = 1e-2
# RATIO_REDUCTION_FACTOR_MIN = 4.0

# MONOTONE_SLACK = 1.1 — SUPERSEDUTA (REV. 3, STATUS.md piano Step 4): questa
# tolleranza moltiplicativa del 10% per passo, applicata al rapporto
# calcolato su TUTTE le shell (incluso il bordo esterno), tollerava una
# sequenza sistematicamente crescente senza discriminarla da una vera
# convergenza rumorosa. Mantenuta qui solo per tracciabilita' storica, NON
# piu' usata dal test 3.5.
MONOTONE_SLACK = 1.1

X_C_SANITY_MAX = 0.1

# --- Costanti REV. 3 (K_BOUNDARY_EXCLUDE ecc.) — SUPERSEDUTE (REV. 4) ------
#
# La REV. 3 escludeva un numero fisso di shell di bordo dal residuo di
# convergenza principale (K_BOUNDARY_EXCLUDE=2) e usava un assert di
# monotonicita' dedicato (MONOTONE_TOL_RTOL). La REV. 4 (STATUS.md, "Piano
# Step 4 rivisto dopo indagine forense") ha sostituito questo approccio con
# una metrica AGGREGATA pesata in massa calcolata su TUTTE le shell (nessuna
# esclusione manuale per indice) — vedi test 3.5 sotto. Le costanti seguenti
# NON sono piu' usate nella logica del test, mantenute solo come nota
# storica per tracciabilita':
# K_BOUNDARY_EXCLUDE = 2
# RATIO_BOUNDARY_SANITY_MAX = 1.0
# MONOTONE_TOL_RTOL = 0.01

N_SHELLS_DEMO_LIKE = 100  # usato nei test 3.2-3.4 (profilo s20)
N_SHELLS_TEST_DEFAULT = 50
N_CONVERGENCE_LIST = [50, 100, 200, 400]
N_POINTS_LANE_EMDEN_CONVERGENCE = 50000

YE_TEST = 0.5

# M_TEST_MSUN / M_SUB_CHANDRA_TEST_MSUN — REV. 4 (STATUS.md, correzione 1):
# ridotte di un fattore 10 (rispetto a 0.01/0.005, REV. 2/3) per portare
# x_c da ~0.0593 a ~0.0128, riducendo il mismatch fisico residuo fra
# l'equilibrio di test (costruito con pressure_non_relativistic) e la
# dinamica (che usa la EOS esatta pressure_chandrasekhar) di un ordine di
# grandezza (Scoperta 1, REV. 4). Rapporto 0.5 fra le due costanti
# invariato rispetto a REV. 2/3.
M_TEST_MSUN = 0.001
M_SUB_CHANDRA_TEST_MSUN = 0.0005

# --- Costanti REV. 4 (metrica aggregata pesata in massa, test 3.5) ---------
#
# global_ratio = sqrt(sum(delta_m_i*a_net_i**2)) / sqrt(sum(delta_m_i*a_grav_i**2))
# calcolato su TUTTE le shell, nessuna esclusione manuale per indice.
#
# Valori EMPIRICI osservati dal coder (M_TEST_MSUN=0.001, YE_TEST=0.5,
# N_POINTS_LANE_EMDEN_CONVERGENCE=50000), calcolati PRIMA di fissare le
# tolleranze sotto, come richiesto dal piano REV. 4:
#   N= 50  -> global_ratio = 4.682491e-02
#   N=100  -> global_ratio = 3.420672e-02
#   N=200  -> global_ratio = 2.475244e-02
#   N=400  -> global_ratio = 1.779589e-02
# (x_c=1.277928e-02, < X_C_SANITY_MAX=0.1, regime non-relativistico atteso)
# Riduzione N=50->N=400: fattore ~2.63. Sequenza monotona decrescente,
# variazione per raddoppio di N: x0.7305, x0.7236, x0.7190 (in lento
# avvicinamento a 1/sqrt(2)~0.7071 — vedi nota sotto).
#
# SCOPERTA NON ANTICIPATA DAL PIANO REV. 4 (da riportare esplicitamente):
# la premessa del piano ("per costruzione [la metrica] sopprime il
# contributo delle shell di bordo... mentre il bulk interno... domina la
# somma") NON e' confermata numericamente. Analisi diagnostica dei
# contributi individuali a N=400 (script separato, non incluso in questa
# suite): il 99.77% del numeratore sum(delta_m_i*a_net_i**2) proviene dalle
# ULTIME 3 shell (indici 397-399, bordo libero esterno), con l'83% dalla
# sola ultima shell (ratio_i~0.53, in linea con il plateau teorico noto
# ~0.5656 del bordo libero, gia' documentato in dynamics.py). Il contributo
# delle shell centrali (incluso il plateau ~0.056-0.058 del centro,
# Scoperta 2 REV. 4) e del bulk interno (che converge correttamente,
# Scoperta 1 REV. 4) e' trascurabile (<0.3% combinato). In altre parole:
# global_ratio a N=400 NON misura principalmente la convergenza del bulk
# interno, ma e' quasi interamente il plateau di bordo esterno gia' noto
# (~0.5656), diluito dal peso di massa O(1/N) di UNA sola shell — questo
# spiega perche' la sequenza osservata scala come ~O(N^-1/2) (dominata da
# un singolo termine di peso ~1/N e ratio costante) invece di convergere
# rapidamente verso il floor fisico atteso (~6e-5) ipotizzato nel piano.
# Il metro resta comunque un test di sanita' valido (verifica che il
# residuo pesato diminuisca con N in modo consistente, non arbitrario), ma
# NON isola la convergenza del bulk come inteso nel piano REV. 4 — questa
# distinzione va riportata al planner/reviewer, non e' stata "aggiustata"
# nascondendola.
#
# GLOBAL_RATIO_TOL_N400: il piano propone 1e-3 come punto di partenza, ma il
# valore osservato a N=400 (1.779589e-02) e' ~18x superiore, per il motivo
# quantificato sopra (dominanza del bordo esterno, non del floor fisico del
# bulk). Fissato empiricamente a 2.5e-2, margine ~1.4x sopra il valore
# osservato (sufficiente ad assorbire piccola variabilita' numerica fra
# piattaforme/versioni scipy, senza svuotare il test: un regressione dello
# schema che alterasse sostanzialmente il residuo di bordo o introducesse
# rumore addizionale nel bulk farebbe comunque fallire l'assert).
GLOBAL_RATIO_TOL_N400 = 2.5e-2

# GLOBAL_RATIO_REDUCTION_FACTOR_MIN: il piano propone 3.0, ma il fattore di
# riduzione osservato N=50->N=400 e' 2.631 (< 3.0), coerente con la
# dominanza del bordo esterno diagnosticata sopra: un singolo termine con
# peso O(1/N) e ratio costante da' una metrica aggregata che scala come
# O(N^-1/2) (fattore atteso per un aumento di N di 8x: sqrt(8)~2.83, in
# linea con l'osservato 2.631), non la riduzione piu' rapida (O(1/N) o
# meglio) che ci si aspetterebbe da una convergenza dominata dal bulk.
# Fissato empiricamente a 2.3, margine ~13% sotto il valore osservato.
GLOBAL_RATIO_REDUCTION_FACTOR_MIN = 2.3


def _s20_profile():
    """Profilo fisico di s20 (Lane-Emden n=2.0), usato da piu' test."""
    catalogo = load_reference_catalog()
    stella = get_progenitor_by_id(catalogo, "s20")
    solution = solve_lane_emden(n=stella.n_politropico)
    profile = physical_profile(solution, stella.densita_centrale_gcm3, stella.massa_nucleo_msun)
    return stella, profile


# --- 3.2, 3.3, 3.4: conservazione della massa e monotonicita' dei raggi ----


def test_build_initial_shells_mass_conservation_internal():
    """sum(delta_m_g) deve combaciare con M_r_totale RICALCOLATO nel test
    con la stessa formula di trapezio (non importato come privato) — per
    essere un controllo di conservazione realmente indipendente
    dall'implementazione interna specifica.
    """
    _, profile = _s20_profile()
    delta_m_g, _, _ = build_initial_shells(profile.r_cm, profile.rho_gcm3, N_SHELLS_DEMO_LIKE)

    from scipy.integrate import cumulative_trapezoid

    m_r_totale = cumulative_trapezoid(
        4.0 * np.pi * profile.r_cm**2 * profile.rho_gcm3, profile.r_cm, initial=0.0
    )[-1]

    assert abs(np.sum(delta_m_g) - m_r_totale) / m_r_totale < TOL_MASS_SHELLS_INTERNAL_RTOL


def test_build_initial_shells_mass_conservation_external():
    stella, profile = _s20_profile()
    delta_m_g, _, _ = build_initial_shells(profile.r_cm, profile.rho_gcm3, N_SHELLS_DEMO_LIKE)

    massa_nominale_g = stella.massa_nucleo_msun * M_SUN_G
    assert abs(np.sum(delta_m_g) - massa_nominale_g) / massa_nominale_g < TOL_MASS_SHELLS_EXTERNAL_RTOL


def test_build_initial_shells_radii_monotonic_and_surface_match():
    _, profile = _s20_profile()
    delta_m_g, m_enclosed_g, r0_cm = build_initial_shells(profile.r_cm, profile.rho_gcm3, N_SHELLS_DEMO_LIKE)

    assert np.all(np.diff(r0_cm) > 0)
    assert abs(r0_cm[-1] - profile.R_cm) / profile.R_cm < TOL_R_LAST_RTOL
    assert np.allclose(m_enclosed_g, np.cumsum(delta_m_g), rtol=1e-12)


# --- 3.5: convergenza dell'equilibrio autoconsistente -----------------------


def test_polytrope_equilibrium_initial_acceleration_convergence():
    """Convergenza dello schema di discretizzazione (REV. 4, STATUS.md,
    "Piano Step 4 rivisto dopo indagine forense") — metrica AGGREGATA
    pesata in massa, calcolata su TUTTE le shell (nessuna esclusione
    manuale per indice, a differenza di REV. 3):

        global_ratio = sqrt(sum(delta_m_i*a_net_i**2)) / sqrt(sum(delta_m_i*a_grav_i**2))

    deve diminuire con N e scendere sotto GLOBAL_RATIO_TOL_N400 a N=400,
    con una riduzione di almeno GLOBAL_RATIO_REDUCTION_FACTOR_MIN fra N=50
    e N=400.

    NOTA (K_NR vs K_UR, planner): `collasso.eos` non espone una
    `k_non_relativistic(ye)` dedicata. Dato che
    pressure_non_relativistic(rho, ye) = K_NR(ye)*rho^(5/3) esattamente
    (il fattore rho^(5/3) si fattorizza), valutare la funzione a rho=1.0
    g/cm^3 restituisce ESATTAMENTE K_NR(ye) — NON confondere con
    k_ultrarelativistic (quella e' per n=3, P_UR∝rho^(4/3), usata da
    chandrasekhar_mass_msun).

    Il controllo su `_shell_dvdt` (import esplicito del privato, stesso
    precedente di `_gamma1_of_x` in Step 3) e' un controllo di
    CONVERGENZA della discretizzazione, non una validazione fisica
    indipendente della formula di accelerazione (stesso principio del
    punto 3.7 di Step 2).

    STORIA (REV. 2/REV. 3, superseduta): le versioni precedenti di questo
    test escludevano un numero fisso di shell di bordo dal residuo
    principale (K_BOUNDARY_EXCLUDE) e verificavano `ratio_interior_max`
    contro RATIO_TOL_N400/RATIO_REDUCTION_FACTOR_MIN. La REV. 3 aveva
    scoperto che, anche escludendo il bordo esterno, il massimo restava
    dominato dalla shell PIU' INTERNA (indice 0, il centro), il cui
    rapporto |a_net|/a_grav non converge a zero ma a un plateau non nullo
    (~0.054-0.056) — un secondo limite strutturale, distinto da quello gia'
    noto del bordo libero esterno (~0.5656). La REV. 4 ha confermato
    quantitativamente (Scoperta 2, STATUS.md) che questo plateau del centro
    e' un effetto della DISCRETIZZAZIONE (non della fisica: persiste anche
    sostituendo pressure_chandrasekhar con la stessa pressure_non_relativistic
    usata per l'equilibrio) e ha diagnosticato un mismatch fisico separato
    (Scoperta 1) fra l'equilibrio di test e la dinamica, dominante nel bulk
    interno per la vecchia scala di massa (x_c~0.0593). Entrambi i risultati
    hanno motivato la correzione REV. 4: M_TEST_MSUN ridotta di 10x (vedi
    sopra) per il mismatch del bulk, e la metrica aggregata pesata in massa
    (sotto) al posto dell'esclusione manuale per indice.

    VALORI EMPIRICI OSSERVATI DAL CODER (REV. 4, calcolati PRIMA di fissare
    GLOBAL_RATIO_TOL_N400/GLOBAL_RATIO_REDUCTION_FACTOR_MIN, come richiesto
    dal piano; M_TEST_MSUN=0.001, YE_TEST=0.5,
    N_POINTS_LANE_EMDEN_CONVERGENCE=50000, x_c=1.277928e-02):

        N= 50  -> global_ratio = 4.682491e-02
        N=100  -> global_ratio = 3.420672e-02
        N=200  -> global_ratio = 2.475244e-02
        N=400  -> global_ratio = 1.779589e-02

    Sequenza monotona decrescente; riduzione N=50->N=400: fattore ~2.631
    (variazione per raddoppio di N: x0.7305, x0.7236, x0.7190, in lento
    avvicinamento a 1/sqrt(2)~0.7071).

    SCOPERTA NON ANTICIPATA DAL PIANO REV. 4 (da riportare esplicitamente,
    non silenziata): la premessa del piano — "per costruzione [la metrica]
    sopprime il contributo delle shell di bordo (centro E superficie), che
    pesano O(1/N) in massa ciascuna, mentre il bulk interno... domina la
    somma" — NON e' confermata numericamente. Analisi diagnostica dei
    contributi individuali al numeratore sum(delta_m_i*a_net_i**2) a N=400
    (script diagnostico separato, non incluso in questa suite): il 99.77%
    del numeratore proviene dalle ULTIME 3 shell (indici 397-399, bordo
    libero esterno), con l'83% dalla sola ultima shell (ratio puntuale
    locale ~0.53, in linea col plateau teorico noto ~0.5656 del bordo
    libero esterno, gia' documentato in `collasso/dynamics.py`). Il
    contributo del centro (shell 0, plateau ~0.057) e del bulk interno
    (che converge correttamente, Scoperta 1 REV. 4) e' <0.3% combinato del
    numeratore. In altre parole: global_ratio NON isola principalmente la
    convergenza del bulk interno come inteso dal piano REV. 4, ma e' quasi
    interamente il plateau di bordo esterno gia' noto (~0.5656), diluito
    dal peso di massa O(1/N) di poche shell — questo spiega la scala di
    convergenza osservata ~O(N^-1/2) (dominanza di un singolo termine con
    peso ~1/N e ratio puntuale costante), molto piu' lenta del floor fisico
    ~6e-5 ipotizzato nel piano come riferimento per GLOBAL_RATIO_TOL_N400.
    La metrica resta comunque un test di sanita' valido (verifica che il
    residuo pesato diminuisca con N in modo consistente e riproducibile,
    non arbitrario/rumoroso), ma la sua interpretazione fisica va
    corretta rispetto al piano REV. 4 — riportato qui per trasparenza,
    non "aggiustato" nascondendo la discrepanza.

    TOLLERANZE: il piano REV. 4 propone GLOBAL_RATIO_TOL_N400=1e-3 e
    GLOBAL_RATIO_REDUCTION_FACTOR_MIN=3.0 come punti di partenza, ma
    autorizza il coder a proporre valori diversi se i numeri osservati lo
    richiedono. I valori osservati sopra (N=400: 1.78e-2; fattore di
    riduzione: 2.63) sono rispettivamente ~18x superiore e ~12% inferiore
    ai punti di partenza del piano, per il motivo quantificato sopra
    (dominanza del bordo esterno). Fissati empiricamente a
    GLOBAL_RATIO_TOL_N400=2.5e-2 (margine ~1.4x sopra l'osservato) e
    GLOBAL_RATIO_REDUCTION_FACTOR_MIN=2.3 (margine ~13% sotto l'osservato)
    — vedi commenti alle costanti in testa al file per il dettaglio della
    motivazione.
    """
    solution = solve_lane_emden(1.5, n_points=N_POINTS_LANE_EMDEN_CONVERGENCE)
    k_nr_cgs = eos.pressure_non_relativistic(1.0, YE_TEST)

    rho_c = polytrope_equilibrium_rho_c_gcm3(
        k_nr_cgs, M_TEST_MSUN * M_SUN_G, solution.xi1, solution.dtheta_dxi_at_xi1
    )

    # Verifica esplicita richiesta dal piano (correzione bloccante 1, REV. 2,
    # invariata in REV. 4): la configurazione deve essere realmente nel
    # regime profondamente non-relativistico assunto dall'equilibrio
    # costruito con pressure_non_relativistic (non solo dichiarato a parole
    # nel piano).
    x_c = eos.fermi_x(rho_c, YE_TEST)
    assert x_c < X_C_SANITY_MAX, f"x_c={x_c} non e' sufficientemente non-relativistico (atteso < {X_C_SANITY_MAX})"

    profile = physical_profile(solution, rho_c, M_TEST_MSUN)

    global_ratios = []
    for n_shells in N_CONVERGENCE_LIST:
        delta_m_g, m_enclosed_g, r0_cm = build_initial_shells(profile.r_cm, profile.rho_gcm3, n_shells)
        a_net = dynamics._shell_dvdt(r0_cm, m_enclosed_g, delta_m_g, YE_TEST)
        a_grav = G_CGS * m_enclosed_g / r0_cm**2

        # global_ratio: metrica aggregata pesata in massa, su TUTTE le
        # shell (nessuna esclusione manuale per indice — REV. 4).
        numeratore = np.sqrt(np.sum(delta_m_g * a_net**2))
        denominatore = np.sqrt(np.sum(delta_m_g * a_grav**2))
        global_ratios.append(numeratore / denominatore)

    # --- Assert di monotonicita' (sequenza osservata pulita, nessun rumore
    # apprezzabile fra i 4 N testati — vedi valori empirici nel docstring). ---
    for i in range(1, len(global_ratios)):
        assert global_ratios[i] < global_ratios[i - 1], (
            f"global_ratio non decrescente fra N={N_CONVERGENCE_LIST[i - 1]} e "
            f"N={N_CONVERGENCE_LIST[i]}: global_ratios={global_ratios}, x_c={x_c}"
        )

    # --- Assert quantitativi principali (REV. 4) ---
    assert global_ratios[-1] < GLOBAL_RATIO_TOL_N400, (
        f"global_ratio a N=400 troppo alto: {global_ratios[-1]} "
        f"(atteso < {GLOBAL_RATIO_TOL_N400}, x_c={x_c})"
    )
    assert global_ratios[0] / global_ratios[-1] > GLOBAL_RATIO_REDUCTION_FACTOR_MIN, (
        f"riduzione insufficiente fra N=50 e N=400: global_ratios={global_ratios}, "
        f"fattore osservato={global_ratios[0] / global_ratios[-1]}, "
        f"atteso > {GLOBAL_RATIO_REDUCTION_FACTOR_MIN}, x_c={x_c}"
    )


# --- 3.6: caso super-Chandrasekhar (s20, collasso atteso) -------------------


def test_super_chandrasekhar_triggers_collapse():
    stella, profile = _s20_profile()
    delta_m_g, _, r0_cm = build_initial_shells(profile.r_cm, profile.rho_gcm3, N_SHELLS_TEST_DEFAULT)

    t_max_s = T_MAX_FREE_FALL_MULTIPLIER * free_fall_time_s(stella.densita_centrale_gcm3)
    result = simulate_collapse(delta_m_g, r0_cm, ye=stella.ye, t_max_s=t_max_s)

    assert result.collapsed is True
    assert result.t_collapse_s is not None and 0 < result.t_collapse_s <= t_max_s
    assert result.rho_c_gcm3_t[-1] > result.rho_c_gcm3_t[0]
    assert result.collapse_reason in ("r_min_threshold", "shell_crossing")


# --- 3.7: caso sub-Chandrasekhar corretto (nessun collasso atteso) ---------


def test_sub_chandrasekhar_does_not_trigger_collapse():
    """Politropo n=3/2 autoconsistente, molto sotto la soglia di
    Chandrasekhar (M_SUB_CHANDRA_TEST_MSUN=0.0005, REV. 4, STATUS.md,
    correzione 5 — ~1/2912 di chandrasekhar_mass_msun(0.5)~1.4559 Msun).
    Atteso nessun collasso entro t_max_s = 10*t_ff.

    Verifica empirica richiesta dal piano (correzione consigliata 4, REV. 2,
    riconfermata dopo l'aggiornamento REV. 4 del valore di
    M_SUB_CHANDRA_TEST_MSUN — correzione 5): se questo test fallisse per un
    falso collasso residuo, occorre prima verificare (a) x_c < X_C_SANITY_MAX,
    (b) l'andamento di rho_c_gcm3_t nel tempo (deve restare sostanzialmente
    piatto). Solo se il problema persistesse si potrebbe ridurre
    T_MAX_FREE_FALL_MULTIPLIER SOLO per questo test — non e' stato
    necessario: il test passa con il valore di default anche col nuovo
    M_SUB_CHANDRA_TEST_MSUN=0.0005 (rieseguito e confermato dal coder, vedi
    esito riportato nel report del ciclo).
    """
    solution = solve_lane_emden(1.5, n_points=N_POINTS_LANE_EMDEN_CONVERGENCE)
    k_nr_cgs = eos.pressure_non_relativistic(1.0, YE_TEST)

    rho_c = polytrope_equilibrium_rho_c_gcm3(
        k_nr_cgs, M_SUB_CHANDRA_TEST_MSUN * M_SUN_G, solution.xi1, solution.dtheta_dxi_at_xi1
    )
    x_c = eos.fermi_x(rho_c, YE_TEST)
    assert x_c < X_C_SANITY_MAX, f"x_c={x_c} non e' sufficientemente non-relativistico (atteso < {X_C_SANITY_MAX})"

    profile = physical_profile(solution, rho_c, M_SUB_CHANDRA_TEST_MSUN)
    delta_m_g, _, r0_cm = build_initial_shells(profile.r_cm, profile.rho_gcm3, N_SHELLS_TEST_DEFAULT)

    t_max_s = T_MAX_FREE_FALL_MULTIPLIER * free_fall_time_s(rho_c)
    result = simulate_collapse(delta_m_g, r0_cm, ye=YE_TEST, t_max_s=t_max_s)

    assert result.collapsed is False, (
        f"falso collasso rilevato: x_c={x_c}, "
        f"rho_c_gcm3_t[0]={result.rho_c_gcm3_t[0]:.6e}, "
        f"rho_c_gcm3_t[-1]={result.rho_c_gcm3_t[-1]:.6e}, "
        f"collapse_reason={result.collapse_reason}"
    )
    assert result.t_collapse_s is None
    assert result.collapse_reason is None


# --- 3.8, 3.9: test aggiuntivi (buona pratica) ------------------------------


def test_free_fall_time_scaling():
    """t_ff proporzionale a rho^-1/2: t_ff(4*rho)/t_ff(rho) = 0.5."""
    rho = 1e9
    assert abs(free_fall_time_s(4 * rho) / free_fall_time_s(rho) - 0.5) < 1e-10


def test_invalid_inputs_raise():
    stella, profile = _s20_profile()
    delta_m_g, _, r0_cm = build_initial_shells(profile.r_cm, profile.rho_gcm3, N_SHELLS_TEST_DEFAULT)

    with pytest.raises(ValueError):
        free_fall_time_s(0.0)
    with pytest.raises(ValueError):
        free_fall_time_s(-1.0)

    with pytest.raises(ValueError):
        build_initial_shells(profile.r_cm, profile.rho_gcm3, 1)
    with pytest.raises(ValueError):
        build_initial_shells(profile.r_cm[:-1], profile.rho_gcm3, N_SHELLS_TEST_DEFAULT)

    with pytest.raises(ValueError):
        simulate_collapse(delta_m_g, r0_cm, ye=stella.ye, t_max_s=0.0)
    with pytest.raises(ValueError):
        simulate_collapse(delta_m_g, r0_cm, ye=stella.ye, t_max_s=1.0, r_min_frac=0.0)
    with pytest.raises(ValueError):
        simulate_collapse(delta_m_g, r0_cm, ye=stella.ye, t_max_s=1.0, r_min_frac=1.0)
    with pytest.raises(ValueError):
        simulate_collapse(delta_m_g[:-1], r0_cm, ye=stella.ye, t_max_s=1.0)


# --- 3.10: evento di shell crossing e floor numerico ------------------------


def test_shell_crossing_event_and_v_floor():
    """Unit test diretto dei nuovi meccanismi REV. 2 (correzione bloccante
    2): evento _shell_crossing_event e floor V_FLOOR_CM3 dentro
    _shell_dvdt.
    """
    n = 4

    # Argomenti extra fittizi (m_enclosed_g, delta_m_g, ye, n, r_min_cm):
    # _shell_crossing_event usa solo n, ma la firma e' allineata a quella
    # comune richiesta da scipy.integrate.solve_ivp per fun ed eventi (vedi
    # nota in collasso.dynamics._collapse_rhs) — gli altri parametri sono
    # ignorati dalla funzione, quindi valori placeholder sono sufficienti
    # per questo unit test diretto.
    delta_m_g_dummy = np.array([1.0e27, 1.0e27, 1.0e27, 1.0e27])
    m_enclosed_g_dummy = np.cumsum(delta_m_g_dummy)
    extra_args = (m_enclosed_g_dummy, delta_m_g_dummy, YE_TEST, n, 1.0e5)

    # Stato sintetico con raggi di bordo ben separati e crescenti.
    r_separati = np.array([1.0e6, 2.0e6, 3.5e6, 5.0e6])
    v_separati = np.zeros(n)
    y_separati = np.concatenate([r_separati, v_separati])
    gap_atteso = np.min(np.diff(np.concatenate(([0.0], r_separati))))
    valore_evento = dynamics._shell_crossing_event(0.0, y_separati, *extra_args)
    assert valore_evento > 0
    assert abs(valore_evento - gap_atteso) < 1e-6

    # Stato sintetico con due raggi di bordo consecutivi resi
    # artificialmente uguali (shell collassata a spessore nullo).
    r_collassati = np.array([1.0e6, 2.0e6, 2.0e6, 5.0e6])
    y_collassati = np.concatenate([r_collassati, v_separati])
    valore_evento_collassato = dynamics._shell_crossing_event(0.0, y_collassati, *extra_args)
    assert abs(valore_evento_collassato) < 1e-6

    # _shell_dvdt con un volume di shell forzato a zero (raggi di bordo
    # consecutivi uguali, passati direttamente alla funzione bypassando
    # simulate_collapse): non deve sollevare eccezione, e il risultato deve
    # essere finito (il floor V_FLOOR_CM3 lo rende numericamente sicuro).
    delta_m_g = np.array([1.0e27, 1.0e27, 1.0e27, 1.0e27])
    m_enclosed_g = np.cumsum(delta_m_g)
    result = dynamics._shell_dvdt(r_collassati, m_enclosed_g, delta_m_g, YE_TEST)
    assert np.all(np.isfinite(result))
    assert V_FLOOR_CM3 == 1e-30
