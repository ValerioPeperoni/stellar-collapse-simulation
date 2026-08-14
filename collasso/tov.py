"""
collasso.tov — integrazione dell'equazione di Tolman-Oppenheimer-Volkoff
(TOV) per una stella di neutroni degeneri, ricerca del limite di massa di
Oppenheimer-Volkoff (Step 6; esteso nel Retrofit Step 6 — EOS realistica).

Riferimento fisico: Oppenheimer & Volkoff, Phys. Rev. 55, 374 (1939). EOS di
neutroni liberi non interagenti (`collasso.eos_neutron`, Step 6) — limite di
modello dichiarato esplicitamente li' (nessuna interazione nucleare forte).
Retrofit Step 6: EOS realistica a politropi a tratti (`collasso.eos_piecewise_polytrope`,
Read et al. 2009, fit SLy/APR4).

*** CORREZIONE DI BUG FISICO (Retrofit Step 6, vedi STATUS.md, sezione
"Retrofit Step 6 — EOS realistica", "CORREZIONE POST-REVIEW") ***: le due
equazioni TOV richiedono come sorgente la densita' di ENERGIA totale
(riposo + interna/relativistica), non la densita' di massa a riposo `rho`.
La versione originale di Step 6 usava erroneamente `rho` sia in `dP/dr` sia
in `dm/dr`. Verificato quantitativamente dall'orchestratore risolvendo la
STESSA EOS a neutroni liberi con le due sorgenti: `rho` (come implementato
in Step 6) da' M_max=0.8427 Msun, `epsilon/c^2` (fisicamente corretto) da'
M_max=0.7100 Msun — quest'ultimo entro ~1.4% dal valore storico di
Oppenheimer & Volkoff 1939 (~0.7 Msun), contro uno scarto del ~20% del
valore bacato. Il path `eos=None` (gas di neutroni liberi) di questo modulo
e' stato CORRETTO per usare `epsilon/c^2` come sorgente (funzione `_tov_rhs`,
sotto) — la vecchia formula (rho come sorgente) resta disponibile SOLO come
riferimento storico/didattico, chiaramente etichettata "non fisicamente
corretta, preservata per confronto" (`_tov_rhs_legacy_bug_rho_source`,
`solve_tov_legacy_bug_rho_source`, `find_ov_mass_limit_legacy_bug_rho_source`),
usata unicamente dallo script demo per mostrare onestamente l'effetto del
fix separato dalla fisica nucleare realistica (SLy/APR4).

Equazioni TOV (forma fisicamente corretta, con densita' di energia epsilon
come sorgente):

    dP/dr = -G*(epsilon/c^2+P/c^2)*(m+4*pi*r^3*P/c^2) / (r^2*(1-2*G*m/(r*c^2)))
    dm/dr = 4*pi*r^2*epsilon/c^2

Due parametrizzazioni della variabile di stato ausiliaria, a seconda del
path:
- `eos=None` (gas di neutroni liberi, Step 6): parametrizzata in x
  (parametro di Fermi adimensionale), per riusare la forma chiusa di dP/dx
  (`collasso.eos_neutron.dP_dx_neutron`) invece di invertire numericamente
  l'EOS ad ogni passo: `dx/dr = (dP/dr) / (dP/dx)`.
- `eos=PiecewisePolytropeEOS` (Retrofit Step 6, SLy/APR4): parametrizzata
  direttamente in rho, riusando la forma chiusa `dP/drho` (`collasso.
  eos_piecewise_polytrope.dP_drho`): `drho/dr = (dP/dr) / (dP/drho)`.

Stato integrato: y = (m, x) oppure y = (m, rho) a seconda del path,
variabile indipendente r (da r0_cm, innesco vicino al centro, a R_cm, la
superficie, individuata da un evento terminale).

Vincoli di modello per questo step (da CLAUDE.md, non negoziabili):
- Gravita': relativistica completa (equazione TOV esatta), NON
  un'approssimazione post-Newtoniana come Step 5 (quella resta limitata al
  termine sorgente di gravita' della dinamica del collasso, non tocca
  questo modulo).
- Nessun termine di raffreddamento neutrini, nessuna rotazione/campo
  magnetico (limiti gia' esclusi a monte, invariati dagli step precedenti).
- Nessuna interazione nucleare forte nell'EOS del gas di neutroni liberi
  (limite dichiarato in `collasso.eos_neutron`, non "corretto" qui); SLy/APR4
  sono fit fenomenologiche (Read et al. 2009), non EOS nucleari a molti
  corpi complete — accurate entro ~1% sulle relazioni massa-raggio.

Nota critica sul RHS (`_tov_rhs`, vedi anche `collasso.dynamics`/
`collasso.relativistic`, Step 4/5, stesso principio gia' stabilito piu'
volte in questo progetto): `_tov_rhs` NON deve MAI sollevare un'eccezione,
perche' e' sul path del RHS passato a `scipy.integrate.solve_ivp`, che non
cattura eccezioni sollevate al suo interno durante le valutazioni di prova
di uno stadio RK45 (puo' valutare il RHS in stati che overshoot oltre una
soglia fisica prima che un evento terminale scatti). Per questo motivo il
RHS applica DUE clamp INCONDIZIONATI (mai una guardia con `ValueError`):
    1. `X_FLOOR_RHS` su `x` (anche se transitoriamente negativo), prima di
       valutare `chandrasekhar_f(x)`/`dP_dx_neutron(x)`/`rho_of_x_neutron(x)`
       — la EOS riusata (`collasso.eos.chandrasekhar_f`) ha una guardia
       rigida che solleva `ValueError` per x<0.
    2. `COMPACTNESS_TOV_CLAMP_MAX` sul termine di compattezza
       `2*G*m/(r*c^2)`, prima di valutare il denominatore
       `1-2*G*m/(r*c^2)` di dP/dr — protegge contro un denominatore che si
       avvicina a zero o diventa negativo durante una valutazione di prova
       del solver (la griglia di densita' centrali esplora fino a
       rho_c=1e17 g/cm^3, ben oltre il massimo di massa atteso, cioe' il
       ramo instabile/piu' compatto).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from collasso.constants import C_LIGHT_CGS, G_CGS, M_SUN_G
from collasso.eos import chandrasekhar_f
from collasso.eos_neutron import (
    P0_N_CGS,
    dP_dx_neutron,
    energy_density_of_x_neutron,
    fermi_x_neutron,
    rho_of_x_neutron,
)
from collasso.eos_piecewise_polytrope import PiecewisePolytropeEOS
from collasso.eos_piecewise_polytrope import dP_drho as _pp_dP_drho
from collasso.eos_piecewise_polytrope import energy_density_of_rho as _pp_energy_density_of_rho
from collasso.eos_piecewise_polytrope import pressure_of_rho as _pp_pressure_of_rho

# --- Costanti fissate dal planner (non a discrezione del coder), vedi
#     STATUS.md, piano Step 6, sezione 1 --------------------------------------

# Innesco vicino a r=0 (cm): x(r0)~=x_c (leading order, dP/dr->0 per r->0
# per simmetria, stesso principio di theta'(0)=0 in Lane-Emden),
# m(r0)~=(4/3)*pi*r0^3*rho_c. Verificato dare risultati stabili nel calcolo
# di controllo dell'orchestratore.
R0_CM_DEFAULT = 1.0

# Clamp INCONDIZIONATO su x dentro il RHS (mai una guardia con ValueError):
# collasso.eos.chandrasekhar_f (riusata direttamente) solleva ValueError per
# x<0, quindi senza questo floor un piccolo overshoot negativo di x in un
# passo di prova di RK45 farebbe esplodere un'eccezione non catturata da
# scipy dentro il RHS.
X_FLOOR_RHS = 1e-6

# Clamp INCONDIZIONATO sulla compattezza 2*G*m/(r*c^2) dentro il RHS, PRIMA
# di valutare il denominatore 1-2*G*m/(r*c^2) di dP/dr — vedi nota di modulo
# sopra (bloccante, da review).
COMPACTNESS_TOV_CLAMP_MAX = 0.99

# Soglia dell'evento terminale di superficie (piu' permissiva di
# X_FLOOR_RHS, per restare fuori dalla zona di stiffness prima che l'evento
# scatti — stesso principio del xi0 di Lane-Emden).
X_SURFACE_EVENT = 1e-4

# Ampio margine sopra qualunque raggio di stella di neutroni atteso
# (~10-20 km), verificato nel calcolo di controllo dell'orchestratore.
R_MAX_CM_DEFAULT = 5.0e7

# Griglia di densita' centrali (g/cm^3) per find_ov_mass_limit, verificata
# dal calcolo di controllo dell'orchestratore contenere il massimo di massa
# (fra 3e15 e 1e16 g/cm^3).
RHO_C_GRID_GCM3 = np.logspace(14, 17, 30)

# --- Costanti aggiunte nel Retrofit Step 6 (EOS realistica, piecewise
#     polytrope) --------------------------------------------------------

# Clamp INCONDIZIONATO su rho dentro il RHS del path piecewise-polytrope
# (stesso principio di X_FLOOR_RHS sopra): evita rho<=0 durante un passo di
# prova di RK45, che farebbe sollevare ValueError dentro `collasso.
# eos_piecewise_polytrope` (le cui funzioni pubbliche validano rho>0).
RHO_FLOOR_RHS_CGS = 1e-3

# Soglia dell'evento terminale di superficie per il path piecewise-polytrope
# (parametrizzato in rho, non in x) — fissata dal planner.
RHO_SURFACE_EVENT_CGS = 1.0

# Griglia di densita' centrali (g/cm^3) per find_ov_mass_limit con EOS
# realistiche (SLy/APR4): il picco di M(rho_c) e' verificato
# dall'orchestratore cadere a rho_c~2e15 g/cm^3 (SLy) e ~1.87e15 g/cm^3
# (APR4), entrambi comodamente dentro questa griglia.
RHO_C_GRID_REALISTIC_CGS = np.logspace(14.5, 16, 30)

# Tolleranza di validazione (larga, deliberatamente) contro il valore
# storico di Oppenheimer & Volkoff 1939 (~0.7 Msun) — vedi nota nel piano
# Step 6, punto 1.10: il valore esatto dipende da dettagli implementativi
# non standardizzati in letteratura (r0, soglia dell'evento di superficie,
# tolleranze del solver), l'obiettivo e' catturare errori di ordine di
# grandezza o di formula, non riprodurre la cifra esatta.
M_TOV_SANITY_RANGE_MSUN = (0.4, 1.2)


@dataclass
class TOVSolution:
    """Soluzione numerica dell'equazione TOV per una data densita' centrale.

    Campi:
        rho_c_gcm3: densita' centrale usata per questa soluzione (g/cm^3).
        r_cm: griglia radiale campionata, da r0_cm a R_cm (cm).
        m_g: massa racchiusa m(r) sulla griglia `r_cm` (g).
        R_cm: raggio della superficie (dove scatta l'evento terminale), cm.
        M_g: massa totale racchiusa alla superficie (g).
        M_msun: massa totale in unita' solari.
        x: parametro di Fermi adimensionale x(r) sulla griglia `r_cm` —
            popolato SOLO per il path a gas di neutroni liberi (`eos=None`,
            incluso il path storico/legacy), `None` per il path
            piecewise-polytrope.
        rho_gcm3: densita' rho(r) sulla griglia `r_cm` — popolato SOLO per
            il path piecewise-polytrope (Retrofit Step 6), `None` per il
            path a gas di neutroni liberi.
    """

    rho_c_gcm3: float
    r_cm: np.ndarray
    m_g: np.ndarray
    R_cm: float
    M_g: float
    M_msun: float
    x: np.ndarray | None = None
    rho_gcm3: np.ndarray | None = None


def _tov_rhs(r: float, y: np.ndarray) -> list[float]:
    """RHS del sistema TOV del primo ordine (m, x), variabile indipendente
    r, path a gas di neutroni liberi (`eos=None`) — VERSIONE CORRETTA
    (Retrofit Step 6, vedi nota di bug fisico nel docstring di modulo):
    usa la densita' di ENERGIA totale `epsilon(x)`
    (`collasso.eos_neutron.energy_density_of_x_neutron`) come sorgente in
    `dP/dr` e `dm/dr`, non la densita' di massa a riposo `rho`.

    Non solleva MAI eccezioni (vedi nota critica nel docstring di modulo)
    — applica due clamp incondizionati (X_FLOOR_RHS su x,
    COMPACTNESS_TOV_CLAMP_MAX sulla compattezza) prima di qualunque
    valutazione dell'EOS o del denominatore di dP/dr.
    """
    m, x = y

    # Clamp 1 (incondizionato): floor su x, anche se transitoriamente
    # negativo durante uno stadio di prova di RK45 — mai una ValueError qui.
    x_clamped = max(x, X_FLOOR_RHS)

    p = P0_N_CGS * chandrasekhar_f(x_clamped)
    dp_dx = dP_dx_neutron(x_clamped)
    epsilon = energy_density_of_x_neutron(x_clamped)

    # Clamp 2 (incondizionato, da review Step 6): tetto sulla compattezza
    # prima di valutare il denominatore di dP/dr.
    compactness_raw = 2.0 * G_CGS * m / (r * C_LIGHT_CGS**2)
    compactness_clamped = min(compactness_raw, COMPACTNESS_TOV_CLAMP_MAX)

    dp_dr = (
        -G_CGS
        * (epsilon / C_LIGHT_CGS**2 + p / C_LIGHT_CGS**2)
        * (m + 4.0 * math.pi * r**3 * p / C_LIGHT_CGS**2)
        / (r**2 * (1.0 - compactness_clamped))
    )
    dm_dr = 4.0 * math.pi * r**2 * epsilon / C_LIGHT_CGS**2
    dx_dr = dp_dr / dp_dx

    return [dm_dr, dx_dr]


def _tov_rhs_legacy_bug_rho_source(r: float, y: np.ndarray) -> list[float]:
    """RHS STORICO di Step 6 — *** BUG FISICO NOTO, PRESERVATO SOLO PER
    CONFRONTO NELLO SCRIPT DEMO *** (vedi nota di bug fisico nel docstring
    di modulo, STATUS.md "Retrofit Step 6", "CORREZIONE POST-REVIEW"): usa
    `rho` (densita' di massa a riposo) come sorgente in `dP/dr` e `dm/dr`,
    invece della densita' di energia totale `epsilon/c^2`. NON
    fisicamente corretto (sottostima la gravitazione del gas relativistico
    vicino al picco OV: M_max=0.8427 Msun invece di 0.7100 Msun). NON
    usare per risultati di produzione — usare sempre `_tov_rhs`.
    """
    m, x = y
    x_clamped = max(x, X_FLOOR_RHS)

    rho = rho_of_x_neutron(x_clamped)
    p = P0_N_CGS * chandrasekhar_f(x_clamped)
    dp_dx = dP_dx_neutron(x_clamped)

    compactness_raw = 2.0 * G_CGS * m / (r * C_LIGHT_CGS**2)
    compactness_clamped = min(compactness_raw, COMPACTNESS_TOV_CLAMP_MAX)

    dp_dr = (
        -G_CGS
        * (rho + p / C_LIGHT_CGS**2)
        * (m + 4.0 * math.pi * r**3 * p / C_LIGHT_CGS**2)
        / (r**2 * (1.0 - compactness_clamped))
    )
    dm_dr = 4.0 * math.pi * r**2 * rho
    dx_dr = dp_dr / dp_dx

    return [dm_dr, dx_dr]


def _tov_rhs_piecewise(r: float, y: np.ndarray, eos: PiecewisePolytropeEOS) -> list[float]:
    """RHS del sistema TOV del primo ordine (m, rho), variabile
    indipendente r, path piecewise-polytrope (Retrofit Step 6, SLy/APR4).
    Parametrizzato direttamente in rho (non in x): riusa la forma chiusa
    `dP_drho` dell'EOS per evitare un'inversione numerica ad ogni passo.

    Non solleva MAI eccezioni (stesso principio di `_tov_rhs`): applica il
    clamp incondizionato RHO_FLOOR_RHS_CGS su rho (anche se transitoriamente
    non positivo durante uno stadio di prova di RK45 — le funzioni
    pubbliche di `collasso.eos_piecewise_polytrope` validano rho>0) e il
    clamp COMPACTNESS_TOV_CLAMP_MAX sulla compattezza, prima di qualunque
    valutazione dell'EOS o del denominatore di dP/dr.
    """
    m, rho = y

    # Clamp 1 (incondizionato): floor su rho.
    rho_clamped = max(rho, RHO_FLOOR_RHS_CGS)

    p = _pp_pressure_of_rho(eos, rho_clamped)
    epsilon = _pp_energy_density_of_rho(eos, rho_clamped)
    dp_drho_val = _pp_dP_drho(eos, rho_clamped)

    # Clamp 2 (incondizionato): tetto sulla compattezza.
    compactness_raw = 2.0 * G_CGS * m / (r * C_LIGHT_CGS**2)
    compactness_clamped = min(compactness_raw, COMPACTNESS_TOV_CLAMP_MAX)

    dp_dr = (
        -G_CGS
        * (epsilon / C_LIGHT_CGS**2 + p / C_LIGHT_CGS**2)
        * (m + 4.0 * math.pi * r**3 * p / C_LIGHT_CGS**2)
        / (r**2 * (1.0 - compactness_clamped))
    )
    dm_dr = 4.0 * math.pi * r**2 * epsilon / C_LIGHT_CGS**2
    drho_dr = dp_dr / dp_drho_val

    return [dm_dr, drho_dr]


def _surface_event(r: float, y: np.ndarray) -> float:
    """Evento terminale: x scende sotto X_SURFACE_EVENT (superficie della
    stella), path a gas di neutroni liberi (`eos=None`, incluso il path
    legacy). Stesso pattern di `_theta_zero_event` (Lane-Emden, Step 2):
    direction=-1 (attraversamento decrescente), terminal=True.
    """
    return y[1] - X_SURFACE_EVENT


_surface_event.direction = -1.0
_surface_event.terminal = True


def _surface_event_rho(r: float, y: np.ndarray) -> float:
    """Evento terminale: rho scende sotto RHO_SURFACE_EVENT_CGS (superficie
    della stella), path piecewise-polytrope. Stesso pattern di
    `_surface_event`.
    """
    return y[1] - RHO_SURFACE_EVENT_CGS


_surface_event_rho.direction = -1.0
_surface_event_rho.terminal = True


def solve_tov(
    rho_c_gcm3: float,
    r0_cm: float = R0_CM_DEFAULT,
    r_max_cm: float = R_MAX_CM_DEFAULT,
    n_points: int = 500,
    eos: PiecewisePolytropeEOS | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-6,
) -> TOVSolution:
    """Risolve l'equazione TOV per la densita' centrale `rho_c_gcm3`.

    Parametri:
        rho_c_gcm3: densita' centrale (g/cm^3), deve essere > 0 stretto.
        r0_cm: raggio di innesco vicino al centro (default R0_CM_DEFAULT).
        r_max_cm: limite superiore di integrazione (default
            R_MAX_CM_DEFAULT), usato come t_span finale di `solve_ivp` (la
            superficie e' comunque attesa entro questo limite).
        n_points: numero di punti della griglia di output (default 500).
        eos: se `None` (default), gas di neutroni liberi non interagenti
            (`collasso.eos_neutron`, Step 6), parametrizzato in x, sorgente
            CORRETTA (`epsilon/c^2`, vedi nota di bug fisico nel docstring
            di modulo). Se una `PiecewisePolytropeEOS` (Retrofit Step 6,
            `collasso.eos_piecewise_polytrope.build_sly()`/`build_apr4()`),
            usa il path parametrizzato in rho.
        rtol, atol: tolleranze del solver RK45 (default quelle gia' fissate
            in Step 6), esposte come parametri per il controllo di
            sensibilita' richiesto dal piano del Retrofit Step 6.

    Ritorna:
        `TOVSolution` con la soluzione campionata su `n_points` punti tra
        r0_cm e R_cm (la superficie). Il campo `x` e' popolato per
        `eos=None`, il campo `rho_gcm3` e' popolato per `eos=
        PiecewisePolytropeEOS` (l'altro campo resta `None`).

    Solleva:
        ValueError: se rho_c_gcm3 <= 0.
        RuntimeError: se l'integrazione di `solve_ivp` fallisce
            (`sol.success is False`), riportando `sol.message`; oppure se
            l'evento di superficie non scatta entro `r_max_cm` (non
            dovrebbe mai accadere con R_MAX_CM_DEFAULT, verificato dal
            calcolo di controllo dell'orchestratore — controllo comunque
            obbligatorio, stesso principio del check `sol.success`).
    """
    if rho_c_gcm3 <= 0:
        raise ValueError(f"rho_c_gcm3={rho_c_gcm3} deve essere strettamente positivo.")

    # max_step fissato dal planner esplicitamente rispetto a R_MAX_CM_DEFAULT
    # (costante di modulo), non rispetto al parametro r_max_cm effettivo —
    # stesso pattern di sicurezza gia' usato in collasso.dynamics (Step 4).
    max_step = R_MAX_CM_DEFAULT / 2000.0

    if eos is None:
        # Innesco in serie (leading order) a r0_cm: x(r0)~=x_c, m(r0)~=
        # (4/3)*pi*r0^3*epsilon(x_c)/c^2 (leading order di dm/dr=4*pi*r^2*
        # epsilon/c^2 vicino al centro, coerente con la sorgente corretta
        # usata in _tov_rhs — r0_cm e' comunque cosi' piccolo, 1 cm, che
        # l'effetto numerico di questa scelta e' del tutto trascurabile).
        x_c = float(fermi_x_neutron(rho_c_gcm3))
        epsilon_c = float(energy_density_of_x_neutron(x_c))
        m0_g = (4.0 / 3.0) * math.pi * r0_cm**3 * epsilon_c / C_LIGHT_CGS**2

        sol = solve_ivp(
            fun=_tov_rhs,
            t_span=(r0_cm, r_max_cm),
            y0=[m0_g, x_c],
            method="RK45",
            rtol=rtol,
            atol=atol,
            dense_output=True,
            max_step=max_step,
            events=_surface_event,
        )
    else:
        # Innesco analogo, parametrizzato in rho: rho(r0)~=rho_c, m(r0)~=
        # (4/3)*pi*r0^3*epsilon(rho_c)/c^2.
        rho_c = float(rho_c_gcm3)
        epsilon_c = float(_pp_energy_density_of_rho(eos, rho_c))
        m0_g = (4.0 / 3.0) * math.pi * r0_cm**3 * epsilon_c / C_LIGHT_CGS**2

        sol = solve_ivp(
            fun=lambda r, y: _tov_rhs_piecewise(r, y, eos),
            t_span=(r0_cm, r_max_cm),
            y0=[m0_g, rho_c],
            method="RK45",
            rtol=rtol,
            atol=atol,
            dense_output=True,
            max_step=max_step,
            events=_surface_event_rho,
        )

    # Check esplicito ed incondizionato su sol.success (stesso pattern
    # obbligatorio di Lane-Emden/Step 4), PRIMA di processare l'evento o
    # costruire TOVSolution.
    if not sol.success:
        raise RuntimeError(f"integrazione TOV fallita per rho_c_gcm3={rho_c_gcm3}: {sol.message}")

    if len(sol.t_events[0]) == 0:
        raise RuntimeError(
            f"nessuna superficie trovata (evento di superficie non scattato) "
            f"entro r_max_cm={r_max_cm} per rho_c_gcm3={rho_c_gcm3} (eos={eos})."
        )

    R_cm = float(sol.t_events[0][0])
    M_g = float(sol.y_events[0][0][0])
    M_msun = M_g / M_SUN_G

    r_cm = np.linspace(r0_cm, R_cm, n_points)
    m_g, secondo_stato = sol.sol(r_cm)

    if eos is None:
        return TOVSolution(
            rho_c_gcm3=rho_c_gcm3,
            r_cm=r_cm,
            m_g=m_g,
            R_cm=R_cm,
            M_g=M_g,
            M_msun=M_msun,
            x=secondo_stato,
            rho_gcm3=None,
        )
    else:
        return TOVSolution(
            rho_c_gcm3=rho_c_gcm3,
            r_cm=r_cm,
            m_g=m_g,
            R_cm=R_cm,
            M_g=M_g,
            M_msun=M_msun,
            x=None,
            rho_gcm3=secondo_stato,
        )


def find_ov_mass_limit(
    rho_c_grid_gcm3: np.ndarray = RHO_C_GRID_GCM3,
    eos: PiecewisePolytropeEOS | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-6,
) -> tuple[float, float, list[TOVSolution]]:
    """Trova il limite di massa di Oppenheimer-Volkoff (o il massimo di
    massa per una EOS piecewise-polytrope) su una griglia di densita'
    centrali.

    Risolve `solve_tov` per ogni densita' della griglia, poi seleziona la
    densita' centrale che massimizza la massa totale (`np.argmax` sulla
    sequenza di masse — nessun raffinamento locale richiesto).

    Parametri:
        rho_c_grid_gcm3: griglia di densita' centrali (default
            RHO_C_GRID_GCM3, per il gas di neutroni liberi; usare
            RHO_C_GRID_REALISTIC_CGS per SLy/APR4).
        eos: passato direttamente a `solve_tov` per ogni punto della
            griglia (vedi `solve_tov` per la semantica).
        rtol, atol: passati direttamente a `solve_tov` (controllo di
            sensibilita' alle tolleranze, Retrofit Step 6).

    Ritorna:
        (rho_c_max_gcm3, M_max_msun, sequenza_completa): la densita'
        centrale a cui si raggiunge la massa massima, la massa massima
        (Msun), e la lista completa delle `TOVSolution` per ogni punto
        della griglia (utile per ispezione/plot futuri, Step 7).
    """
    rho_c_grid_gcm3 = np.asarray(rho_c_grid_gcm3, dtype=float)
    sequenza_completa: list[TOVSolution] = []
    masse_msun = np.empty(len(rho_c_grid_gcm3))

    for i, rho_c in enumerate(rho_c_grid_gcm3):
        soluzione = solve_tov(float(rho_c), eos=eos, rtol=rtol, atol=atol)
        sequenza_completa.append(soluzione)
        masse_msun[i] = soluzione.M_msun

    idx_max = int(np.argmax(masse_msun))
    rho_c_max_gcm3 = float(rho_c_grid_gcm3[idx_max])
    M_max_msun = float(masse_msun[idx_max])

    return rho_c_max_gcm3, M_max_msun, sequenza_completa


# --- Path storico/legacy (BUG FISICO, preservato SOLO per confronto) -------


def solve_tov_legacy_bug_rho_source(
    rho_c_gcm3: float,
    r0_cm: float = R0_CM_DEFAULT,
    r_max_cm: float = R_MAX_CM_DEFAULT,
    n_points: int = 500,
) -> TOVSolution:
    """Risolve l'equazione TOV con la formula STORICA (bacata) di Step 6,
    che usa `rho` invece di `epsilon/c^2` come sorgente in `dP/dr` e
    `dm/dr` (vedi nota di bug fisico nel docstring di modulo). Preservata
    SOLO per il confronto a tre vie dello script demo (`scripts/
    step6_demo.py`) — NON usare per risultati di produzione, usare sempre
    `solve_tov` (eos=None), che e' fisicamente corretto.
    """
    if rho_c_gcm3 <= 0:
        raise ValueError(f"rho_c_gcm3={rho_c_gcm3} deve essere strettamente positivo.")

    x_c = float(fermi_x_neutron(rho_c_gcm3))
    # Innesco storico (bacato): usa rho_c invece di epsilon_c/c^2, coerente
    # con la sorgente (bacata) usata in dm/dr da questo path.
    m0_g = (4.0 / 3.0) * math.pi * r0_cm**3 * rho_c_gcm3

    max_step = R_MAX_CM_DEFAULT / 2000.0

    sol = solve_ivp(
        fun=_tov_rhs_legacy_bug_rho_source,
        t_span=(r0_cm, r_max_cm),
        y0=[m0_g, x_c],
        method="RK45",
        rtol=1e-8,
        atol=1e-6,
        dense_output=True,
        max_step=max_step,
        events=_surface_event,
    )

    if not sol.success:
        raise RuntimeError(f"integrazione TOV (legacy) fallita per rho_c_gcm3={rho_c_gcm3}: {sol.message}")

    if len(sol.t_events[0]) == 0:
        raise RuntimeError(
            f"nessuna superficie trovata (path legacy) entro r_max_cm={r_max_cm} "
            f"per rho_c_gcm3={rho_c_gcm3}."
        )

    R_cm = float(sol.t_events[0][0])
    M_g = float(sol.y_events[0][0][0])
    M_msun = M_g / M_SUN_G

    r_cm = np.linspace(r0_cm, R_cm, n_points)
    m_g, x = sol.sol(r_cm)

    return TOVSolution(
        rho_c_gcm3=rho_c_gcm3,
        r_cm=r_cm,
        m_g=m_g,
        R_cm=R_cm,
        M_g=M_g,
        M_msun=M_msun,
        x=x,
        rho_gcm3=None,
    )


def find_ov_mass_limit_legacy_bug_rho_source(
    rho_c_grid_gcm3: np.ndarray = RHO_C_GRID_GCM3,
) -> tuple[float, float, list[TOVSolution]]:
    """Equivalente di `find_ov_mass_limit` ma con il path storico (bacato)
    `solve_tov_legacy_bug_rho_source` — SOLO per il confronto a tre vie
    dello script demo, non per risultati di produzione.
    """
    rho_c_grid_gcm3 = np.asarray(rho_c_grid_gcm3, dtype=float)
    sequenza_completa: list[TOVSolution] = []
    masse_msun = np.empty(len(rho_c_grid_gcm3))

    for i, rho_c in enumerate(rho_c_grid_gcm3):
        soluzione = solve_tov_legacy_bug_rho_source(float(rho_c))
        sequenza_completa.append(soluzione)
        masse_msun[i] = soluzione.M_msun

    idx_max = int(np.argmax(masse_msun))
    rho_c_max_gcm3 = float(rho_c_grid_gcm3[idx_max])
    M_max_msun = float(masse_msun[idx_max])

    return rho_c_max_gcm3, M_max_msun, sequenza_completa
