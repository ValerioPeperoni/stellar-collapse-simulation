"""
collasso.eos_piecewise_polytrope — EOS realistica a politropi a tratti
(piecewise polytrope), parametrizzazione di Read, Lackey, Owen & Friedman
(2009, PRD 79, 124032, arXiv:0812.2163), fit a SLy/APR4 (Retrofit Step 6).

Sostituisce, per la classificazione del remnant, il gas di neutroni liberi
non interagenti di Step 6 (`collasso.eos_neutron`) con una parametrizzazione
fenomenologica che riproduce le relazioni massa-raggio di EOS nucleari
realistiche entro ~1% (Read et al. 2009).

Algoritmo e costanti fissati dal planner/orchestratore (STATUS.md, sezione
"Retrofit Step 6 — EOS realistica"), non a discrezione del coder: crosta
fissa (fit SLy4, Read et al. Tabella II), nucleo a 3 politropi (parametri
SLy/APR4), costruzione a 7 pezzi con continuita' di P e di epsilon ai
confini. Costanti verificate indipendentemente dall'orchestratore contro il
codice sorgente di produzione di LALSuite (LIGO/Virgo,
`lalsimulation/lib/LALSimNeutronStarEOSPiecewisePolytrope.c`, che implementa
letteralmente Read et al. 2009): risultato di controllo SLy -> M_max=2.000
Msun, APR4 -> M_max=2.187 Msun, entro il 2-3% dei valori di letteratura
(SLy~=2.05, APR4~=2.20 Msun).

Vincolo di modello (CLAUDE.md, tabella vincoli): EOS **fredda** (T=0),
nessun termine termico/entropico — stessa ipotesi di `collasso.eos`/
`collasso.eos_neutron`. Fit fenomenologico, NON una EOS nucleare completa a
molti corpi: accurato entro ~1% sulle relazioni massa-raggio (Read et al.
2009), non su ogni osservabile microscopica.

Caso "ponte" a 8 pezzi (per parametri fuori dal range di applicabilita' a 7
pezzi, gestito da LALSuite) NON implementato in questo progetto: se il
calcolo di `rho0` non cade fra `RHO_LOW_CGS[3]` e `RHO_1_CGS`, viene
sollevato un `ValueError` esplicito, mai gestito silenziosamente.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from collasso.constants import C_LIGHT_CGS

# --- Crosta: fit fisso a SLy4 (Read et al. 2009, Tabella II), trascritto da
#     LALSuite, uguale per qualunque EOS ad alta densita' -------------------

RHO_LOW_CGS = [0.0, 2.44033979e7, 3.78358138e11, 2.62780487e12]  # g/cm^3
K_LOW_CGS = [
    6.11252036792443e12,
    9.54352947022931e14,
    4.787640050002652e22,
    3.593885515256112e13,
]
GAMMA_LOW = [1.58424999, 1.28732904, 0.62223344, 1.35692395]

# --- Nucleo: densita' di separazione fisse fra i 3 politropi del nucleo ----
RHO_1_CGS = 10.0**14.7  # g/cm^3
RHO_2_CGS = 10.0**15.0  # g/cm^3

# Parametri di tabella (Read et al. 2009), citazione esplicita: log10(p1) in
# cgs (pressione al confine RHO_1_CGS), indici adiabatici dei 3 politropi
# del nucleo.
SLY = {"log10_p1": 34.348, "gamma1": 3.005, "gamma2": 2.988, "gamma3": 2.851}
APR4 = {"log10_p1": 34.269, "gamma1": 2.830, "gamma2": 3.445, "gamma3": 3.348}

# Valori di letteratura di M_max (Read et al. 2009 e successivi), usati come
# riferimento per i test di plausibilita' (non ricalcolati qui, solo citati).
M_MAX_LITERATURE_MSUN = {"SLy": 2.05, "APR4": 2.20}


def _ensure_scalar_return(arr, original_input):
    """Ritorna arr come float scalare se original_input era scalare,
    altrimenti ndarray (stesso pattern di collasso.eos)."""
    if np.isscalar(original_input) or (isinstance(original_input, np.ndarray) and original_input.ndim == 0):
        return float(arr)
    return arr


def _validate_rho(rho_gcm3):
    """Valida rho_gcm3 > 0 strettamente (scalare o elementwise su ndarray),
    stesso stile di `collasso.eos._validate_rho`."""
    rho_arr = np.asarray(rho_gcm3, dtype=float)
    if np.any(rho_arr <= 0):
        raise ValueError(
            f"rho_gcm3 deve essere strettamente positivo (>0); trovato valore minimo={np.min(rho_arr)}"
        )


@dataclass
class PiecewisePolytropeEOS:
    """EOS a politropi a tratti (7 pezzi: 4 di crosta + 3 di nucleo).

    Campi (tutti np.ndarray di lunghezza 7, indicizzati per pezzo i=0..6):
        nome: identificativo ("SLy", "APR4", ...).
        rho_boundaries: confine INFERIORE di ciascun pezzo (g/cm^3); il
            pezzo i e' attivo per rho in [rho_boundaries[i],
            rho_boundaries[i+1]) per i<6, e per rho>=rho_boundaries[6] per
            i=6 (nessun limite superiore).
        K: costante politropica del pezzo i (unita' cgs, P=K*rho^Gamma).
        Gamma: indice adiabatico del pezzo i.
        n: indice politropico del pezzo i, n_i = 1/(Gamma_i - 1).
        a: costante di integrazione dell'energia del pezzo i (continuita'
            di epsilon ai confini), a_0=0 per costruzione.
        P_boundaries: pressione al confine INFERIORE del pezzo i
            (P_boundaries[0]=0 per costruzione, rho_boundaries[0]=0).
    """

    nome: str
    rho_boundaries: np.ndarray
    K: np.ndarray
    Gamma: np.ndarray
    n: np.ndarray
    a: np.ndarray
    P_boundaries: np.ndarray


def _build_piecewise_polytrope(nome: str, log10_p1: float, gamma1: float, gamma2: float, gamma3: float) -> PiecewisePolytropeEOS:
    """Costruisce la EOS a 7 pezzi per una data terna di parametri del
    nucleo (log10_p1, gamma1, gamma2, gamma3), formule fissate dal planner
    (STATUS.md, Retrofit Step 6):

        p1 = 10**log10_p1
        k1 = p1 / RHO_1_CGS**gamma1
        k2 = p1 / RHO_1_CGS**gamma2
        k3 = k2 * RHO_2_CGS**(gamma2-gamma3)
        rho0 = (K_LOW_CGS[3]/k1) ** (1/(gamma1-GAMMA_LOW[3]))

    Continuita' di P ai confini garantita per costruzione (k1/k2 ancorati a
    p1 in RHO_1_CGS, k3 ancorato a k2 in RHO_2_CGS, rho0 e' il punto in cui
    il quarto pezzo di crosta e il primo pezzo di nucleo danno la stessa
    pressione). Costanti di integrazione dell'energia a_i costruite per
    continuita' di epsilon (prima legge della termodinamica).
    """
    p1 = 10.0**log10_p1
    k1 = p1 / RHO_1_CGS**gamma1
    k2 = p1 / RHO_1_CGS**gamma2
    k3 = k2 * RHO_2_CGS ** (gamma2 - gamma3)
    rho0 = (K_LOW_CGS[3] / k1) ** (1.0 / (gamma1 - GAMMA_LOW[3]))

    if not (RHO_LOW_CGS[3] < rho0 < RHO_1_CGS):
        raise ValueError(
            f"eos_piecewise_polytrope: per '{nome}' (log10_p1={log10_p1}, "
            f"gamma1={gamma1}, gamma2={gamma2}, gamma3={gamma3}) il punto di "
            f"raccordo crosta/nucleo rho0={rho0:.6e} g/cm^3 non cade nel "
            f"range (RHO_LOW_CGS[3]={RHO_LOW_CGS[3]:.6e}, "
            f"RHO_1_CGS={RHO_1_CGS:.6e}) g/cm^3 richiesto dalla costruzione "
            "a 7 pezzi di Read et al. (2009). Il caso 'ponte' a 8 pezzi di "
            "LALSuite per parametri fuori range non e' implementato in "
            "questo progetto: nessuna gestione silenziosa, errore esplicito."
        )

    rho_boundaries = np.array(
        [0.0, RHO_LOW_CGS[1], RHO_LOW_CGS[2], RHO_LOW_CGS[3], rho0, RHO_1_CGS, RHO_2_CGS],
        dtype=float,
    )
    K = np.array([*K_LOW_CGS, k1, k2, k3], dtype=float)
    Gamma = np.array([*GAMMA_LOW, gamma1, gamma2, gamma3], dtype=float)
    n = 1.0 / (Gamma - 1.0)

    # P al confine inferiore di ciascun pezzo, con le costanti PROPRIE del
    # pezzo (P_boundaries[0]=0 automaticamente: rho_boundaries[0]=0 e
    # GAMMA_LOW[0]>0, quindi 0**GAMMA_LOW[0]=0 — nessun caso speciale
    # necessario).
    P_boundaries = K * rho_boundaries**Gamma

    # Costanti di integrazione dell'energia (continuita' di epsilon ai
    # confini, prima legge della termodinamica): a_0=0, a_i = a_(i-1) +
    # (n_(i-1)-n_i)*P_i/(rho_boundary_i*c^2) per i>=1 (rho_boundaries[i]>0
    # per i>=1, nessuna divisione per zero).
    a = np.zeros(7)
    for i in range(1, 7):
        a[i] = a[i - 1] + (n[i - 1] - n[i]) * P_boundaries[i] / (rho_boundaries[i] * C_LIGHT_CGS**2)

    return PiecewisePolytropeEOS(
        nome=nome,
        rho_boundaries=rho_boundaries,
        K=K,
        Gamma=Gamma,
        n=n,
        a=a,
        P_boundaries=P_boundaries,
    )


def build_sly() -> PiecewisePolytropeEOS:
    """EOS a politropi a tratti, fit a SLy (Read et al. 2009, Tabella III)."""
    return _build_piecewise_polytrope("SLy", **SLY)


def build_apr4() -> PiecewisePolytropeEOS:
    """EOS a politropi a tratti, fit a APR4 (Read et al. 2009, Tabella III)."""
    return _build_piecewise_polytrope("APR4", **APR4)


def _active_piece_index(eos: PiecewisePolytropeEOS, rho_arr: np.ndarray) -> np.ndarray:
    """Indice del pezzo attivo (il rho_boundary piu' grande <= rho),
    vettorizzato via np.searchsorted. rho_arr deve essere > 0 (validato dal
    chiamante) — searchsorted su rho_boundaries (che parte da 0) con
    side="right" ritorna sempre indice >=1 per rho>0, quindi idx-1>=0.
    Clip finale a [0, 6] per sicurezza (rho oltre RHO_2_CGS resta sul pezzo
    6, nessun limite superiore per costruzione)."""
    idx = np.searchsorted(eos.rho_boundaries, rho_arr, side="right") - 1
    return np.clip(idx, 0, len(eos.rho_boundaries) - 1)


def pressure_of_rho(eos: PiecewisePolytropeEOS, rho_gcm3):
    """P(rho) = K_i * rho^Gamma_i, pezzo attivo i (erg/cm^3). Vettorizzata."""
    _validate_rho(rho_gcm3)
    rho_arr = np.asarray(rho_gcm3, dtype=float)
    idx = _active_piece_index(eos, rho_arr)
    P = eos.K[idx] * rho_arr ** eos.Gamma[idx]
    return _ensure_scalar_return(P, rho_gcm3)


def energy_density_of_rho(eos: PiecewisePolytropeEOS, rho_gcm3):
    """epsilon(rho) = (1+a_i)*rho*c^2 + n_i*P(rho), pezzo attivo i
    (erg/cm^3). Vettorizzata."""
    _validate_rho(rho_gcm3)
    rho_arr = np.asarray(rho_gcm3, dtype=float)
    idx = _active_piece_index(eos, rho_arr)
    P = eos.K[idx] * rho_arr ** eos.Gamma[idx]
    epsilon = (1.0 + eos.a[idx]) * rho_arr * C_LIGHT_CGS**2 + eos.n[idx] * P
    return _ensure_scalar_return(epsilon, rho_gcm3)


def dP_drho(eos: PiecewisePolytropeEOS, rho_gcm3):
    """dP/drho(rho) = K_i * Gamma_i * rho^(Gamma_i-1), forma chiusa, pezzo
    attivo i. Vettorizzata.

    NOTA: dP/drho e' discontinua (C0 ma non C1) ai 6 confini interni fra
    pezzi politropici — fatto noto e atteso della parametrizzazione a
    politropi a tratti (Read et al. 2009), non un errore. Verificato non
    causare instabilita' apprezzabili nel solver TOV dal controllo di
    sensibilita' alle tolleranze (STATUS.md, Retrofit Step 6).
    """
    _validate_rho(rho_gcm3)
    rho_arr = np.asarray(rho_gcm3, dtype=float)
    idx = _active_piece_index(eos, rho_arr)
    deriv = eos.K[idx] * eos.Gamma[idx] * rho_arr ** (eos.Gamma[idx] - 1.0)
    return _ensure_scalar_return(deriv, rho_gcm3)
