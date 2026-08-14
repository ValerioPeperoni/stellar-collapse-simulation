"""
collasso.eos_neutron — EOS di neutroni degeneri (gas di neutroni liberi
non interagenti), formalismo di Oppenheimer & Volkoff (Phys. Rev. 55, 374,
1939), Step 6.

Stesso formalismo esatto della EOS di elettroni degeneri di Chandrasekhar
(`collasso.eos`, Step 3), con la massa del neutrone al posto della massa
dell'elettrone e densita' numerica di neutroni diretta da rho
(`n_n = rho/M_NEUTRON_G`, NESSUN Ye — a differenza del gas di elettroni
degeneri, qui la materia e' fatta di soli neutroni liberi, non c'e' una
frazione di elettroni da specificare).

Limite di modello dichiarato esplicitamente (CLAUDE.md, tabella vincoli,
riga "Classificazione remnant": limite di Oppenheimer-Volkoff): questa EOS
trascura le interazioni nucleari forti fra neutroni, quindi sottostima
sistematicamente il vero limite di massa osservativo delle stelle di
neutroni (~2-2.2 Msun, es. PSR J0740+6620) — riproduce solo l'ordine di
grandezza storico di Oppenheimer & Volkoff 1939 (~0.7 Msun).

Riuso genuino (non duplicazione): `pressure_degenerate_neutron` chiama
DIRETTAMENTE `collasso.eos.chandrasekhar_f` (import esplicito dal modulo
Step 3). f(x) e' una funzione puramente adimensionale di x (nessuna
dipendenza dalla massa del fermione sottostante), quindi e' corretto e
necessario riusarla identica invece di ricopiarne la logica (branching a
due rami per la cancellazione catastrofica a x piccolo incluso
automaticamente).

Questo modulo NON modifica `collasso/eos.py` (nuovo file separato, come da
piano Step 6).
"""

from __future__ import annotations

import math

import numpy as np

from collasso.constants import C_LIGHT_CGS, HBAR_CGS, M_NEUTRON_G
from collasso.eos import chandrasekhar_f

# Costante ricorrente (densita' numerica di neutroni / parametro di Fermi),
# stessa forma di _3_PI2 in collasso.eos.
_3_PI2 = 3.0 * math.pi**2

# Prefattore della pressione di neutroni degeneri (erg/cm^3 per unita' di
# f(x)): stessa forma di `_P0_CGS` in collasso.eos, con M_NEUTRON_G al posto
# di M_ELECTRON_G. Esposto pubblicamente (senza underscore, a differenza di
# `_P0_CGS`) perche' riusato direttamente da `collasso.tov` dentro il RHS di
# `solve_ivp`, che e' parametrizzato in x (non in rho) — vedi piano Step 6,
# equazioni TOV.
P0_N_CGS = (M_NEUTRON_G**4 * C_LIGHT_CGS**5) / (24.0 * math.pi**2 * HBAR_CGS**3)


def _validate_rho(rho_gcm3):
    """Valida rho_gcm3 > 0 strettamente (scalare o elementwise su ndarray).

    Stesso stile di `collasso.eos._validate_rho`. Usata SOLO dalle funzioni
    pubbliche non sul path del RHS di `solve_ivp` (`fermi_x_neutron`,
    `pressure_degenerate_neutron`) — le funzioni chiamate dentro il RHS di
    `collasso.tov` (`dP_dx_neutron`, `rho_of_x_neutron`) non validano
    l'input e non sollevano mai eccezioni, per lo stesso motivo di
    `_shell_dvdt`/`relativistic_grav_accel` (Step 4/5): un'eccezione dentro
    il RHS non viene catturata da scipy e causa un crash non gestito.
    """
    rho_arr = np.asarray(rho_gcm3, dtype=float)
    if np.any(rho_arr <= 0):
        raise ValueError(
            f"rho_gcm3 deve essere strettamente positivo (>0); trovato "
            f"valore minimo={np.min(rho_arr)}"
        )


def fermi_x_neutron(rho_gcm3):
    """Parametro di Fermi adimensionale del gas di neutroni degeneri.

    x_n(rho) = (hbar/(m_n*c)) * (3*pi^2*n_n)^(1/3), con n_n = rho/M_NEUTRON_G
    (nessun Ye, a differenza di `collasso.eos.fermi_x`).

    Funzione pubblica non sul path del RHS di `solve_ivp`: valida l'input
    (rho_gcm3 > 0 stretto) e puo' sollevare `ValueError` in sicurezza.
    Vettorizzata (scalari o np.ndarray).
    """
    _validate_rho(rho_gcm3)
    rho_arr = np.asarray(rho_gcm3, dtype=float)
    n_n = rho_arr / M_NEUTRON_G
    x = (HBAR_CGS / (M_NEUTRON_G * C_LIGHT_CGS)) * (_3_PI2 * n_n) ** (1.0 / 3.0)
    return x


def pressure_degenerate_neutron(rho_gcm3):
    """Pressione di degenerazione del gas di neutroni liberi (erg/cm^3).

    P(rho) = P0_N_CGS * chandrasekhar_f(x_n(rho)), dove
    `chandrasekhar_f` e' RIUSATA DIRETTAMENTE da `collasso.eos` (import
    esplicito in testa al modulo) — non ne duplica la logica a due rami.

    Funzione pubblica non sul path del RHS di `solve_ivp`: valida l'input
    tramite `fermi_x_neutron` (rho_gcm3 > 0 stretto). Vettorizzata.
    """
    x = fermi_x_neutron(rho_gcm3)
    return P0_N_CGS * chandrasekhar_f(x)


def dP_dx_neutron(x):
    """dP/dx = P0_N_CGS * 8*x^4/sqrt(1+x^2) (identita' esatta, stessa forma
    algebrica di Step 3 — derivata di `chandrasekhar_f`, riusata non
    ri-derivata qui).

    ATTENZIONE: questa funzione e' chiamata da `collasso.tov` dentro il RHS
    passato a `solve_ivp` (parametrizzazione TOV in x). NON valida l'input
    (nessuna guardia con `ValueError`): il chiamante e' responsabile di
    passare `x` gia' clampato a un valore non-negativo (`X_FLOOR_RHS`,
    `collasso.tov`) PRIMA di questa chiamata — stesso principio del vincolo
    su `_shell_dvdt`/`relativistic_grav_accel` (Step 4/5): un'eccezione
    dentro il RHS di `solve_ivp` non viene catturata da scipy.
    """
    x_arr = np.asarray(x, dtype=float)
    return P0_N_CGS * 8.0 * x_arr**4 / np.sqrt(1.0 + x_arr**2)


def energy_density_of_x_neutron(x):
    """Densita' di energia TOTALE (riposo incluso) del gas di neutroni
    liberi degeneri, in funzione del parametro di Fermi x (erg/cm^3).

    epsilon(x) = P0_N_CGS * 3*[x*(2*x^2+1)*sqrt(1+x^2) - asinh(x)]

    Derivata dalla relazione termodinamica esatta epsilon = n*mu - P, con
    mu = m_n*c^2*sqrt(1+x^2) (energia di Fermi relativistica, riposo
    incluso) — formula fissata dal planner/orchestratore (STATUS.md,
    Retrofit Step 6, "CORREZIONE POST-REVIEW"), verificata nel limite
    non-relativistico x->0, dove si riduce correttamente a
    epsilon -> rho(x)*c^2 (sviluppo in serie: epsilon(x) ~ 8*P0_N_CGS*x^3,
    e rho_of_x_neutron(x)*C_LIGHT_CGS^2 = 8*P0_N_CGS*x^3 identicamente,
    per costruzione dei due prefattori).

    ATTENZIONE (stesso status di `dP_dx_neutron`/`rho_of_x_neutron`): questa
    funzione E' chiamata dal RHS di `collasso.tov` (`_tov_rhs`, path
    eos=None, CORRETTO), quindi non valida l'input e non deve mai sollevare
    eccezioni — il chiamante e' responsabile di passare `x` gia' clampato a
    un valore non-negativo (`X_FLOOR_RHS`, `collasso.tov`) prima di questa
    chiamata.
    """
    x_arr = np.asarray(x, dtype=float)
    epsilon = P0_N_CGS * 3.0 * (x_arr * (2.0 * x_arr**2 + 1.0) * np.sqrt(1.0 + x_arr**2) - np.arcsinh(x_arr))
    return epsilon


def energy_density_of_rho_neutron(rho_gcm3):
    """Densita' di energia totale del gas di neutroni liberi degeneri, in
    funzione di rho (erg/cm^3): epsilon(rho) = energy_density_of_x_neutron(
    fermi_x_neutron(rho)).

    Funzione pubblica non sul path del RHS di `solve_ivp`: valida l'input
    tramite `fermi_x_neutron` (rho_gcm3 > 0 stretto). Vettorizzata.
    """
    x = fermi_x_neutron(rho_gcm3)
    return energy_density_of_x_neutron(x)


def rho_of_x_neutron(x):
    """rho(x) = M_NEUTRON_G^4*C_LIGHT_CGS^3*x^3 / (3*pi^2*HBAR_CGS^3),
    inversione in forma chiusa (NON root-finding) di `fermi_x_neutron`,
    formula fissata dal planner.

    Stessa nota di `dP_dx_neutron`: nessuna validazione dell'input, questa
    funzione e' sul path del RHS di `solve_ivp` in `collasso.tov` e non
    deve mai sollevare eccezioni.
    """
    x_arr = np.asarray(x, dtype=float)
    return (M_NEUTRON_G**4 * C_LIGHT_CGS**3 * x_arr**3) / (_3_PI2 * HBAR_CGS**3)
