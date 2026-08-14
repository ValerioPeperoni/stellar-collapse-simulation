"""
collasso.relativistic — correzioni relativistiche approssimate sul termine
sorgente di gravita' del collasso (Step 5).

Riferimento fisico: potenziale gravitazionale effettivo "Case A" di Marek,
Dimmelmeier, Janka, Mueller & Buras (2006, A&A 445, 273) — standard di
riferimento per correzioni relativistiche approssimate in codici Newtoniani
di collasso stellare, derivato dalla riduzione dell'equazione TOV al limite
post-Newtoniano:

    a_grav_GR(r) = -G * [m(r) + 4*pi*r^3*P(r)/c^2] * [1 + P(r)/(rho(r)*c^2)]
                   / (r^2 * [1 - 2*G*m(r)/(r*c^2)])

Riduce esattamente a -G*m/r^2 (Newtoniano, Step 4) quando P/(rho*c^2)->0 e
2*G*m/(r*c^2)->0 (campo debole).

Vincoli di modello per questo step (da CLAUDE.md e piano Step 5, non
negoziabili):
- Si modifica SOLO il termine sorgente di gravita' gia' esistente in
  `_shell_dvdt` (collasso.dynamics, Step 4): il termine di pressione NON e'
  toccato — limite dichiarato esplicitamente, non una correzione GR
  completa dell'idrodinamica.
- Nessun limite TOV/classificazione remnant introdotto qui (Step 6).
- Nessun termine di raffreddamento neutrini, nessuna rotazione/campo
  magnetico (limiti gia' esclusi a monte, invariati da Step 1).

Nota critica (vedi docstring di `relativistic_grav_accel` sotto):
`relativistic_grav_accel` NON deve MAI sollevare un'eccezione, perche' e'
chiamata da `_shell_dvdt`, che e' sul path del RHS passato a
`scipy.integrate.solve_ivp` — stesso principio gia' scoperto e corretto in
Step 4 REV. 2 per `V_FLOOR_CM3`: un'eccezione dentro il RHS non viene
catturata da scipy e causa un crash non gestito del processo, perche' gli
eventi di `solve_ivp` sono controllati fra step accettati, non ad ogni
valutazione interna dello stadio RK45 (il solver puo' valutare il RHS in
stati di prova oltre la soglia prima che l'evento scatti).
"""

from __future__ import annotations

import numpy as np

from collasso.constants import C_LIGHT_CGS, G_CGS

# --- Costanti fissate dal planner (non a discrezione del coder), vedi
#     STATUS.md, piano Step 5, sezione 3 -------------------------------------

# Soglia di sicurezza (evento terminale, non un clamp interno): oltre questa
# compattezza l'approssimazione "Case A" perde comunque di significativita'
# fisica, indipendentemente dal fatto che il denominatore resti positivo.
COMPACTNESS_SAFETY_LIMIT = 0.9

# Clamp numerico interno usato SOLO dentro relativistic_grav_accel come rete
# di sicurezza residua (stesso ruolo di V_FLOOR_CM3 in Step 4/dynamics.py),
# MAI un'eccezione. In pratica non dovrebbe mai essere esercitato, perche'
# l'evento terminale a COMPACTNESS_SAFETY_LIMIT=0.9 ferma l'integrazione
# prima che la compattezza raggiunga 0.99.
COMPACTNESS_CLAMP_MAX = 0.99


def compactness(m_g, r_cm) -> float:
    """Compattezza adimensionale 2*G*m/(r*c^2).

    Guardia: ValueError se r_cm<=0 o m_g<=0 (stesso stile delle guardie gia'
    usate in collasso.dynamics/collasso.eos). Questa funzione NON e' sul
    path del RHS di solve_ivp (e' usata come utility pubblica, es. dagli
    script demo e dai test) — a differenza di relativistic_grav_accel, puo'
    quindi sollevare eccezioni in sicurezza.
    """
    if r_cm <= 0:
        raise ValueError(f"r_cm={r_cm} deve essere strettamente positivo.")
    if m_g <= 0:
        raise ValueError(f"m_g={m_g} deve essere strettamente positivo.")
    return 2.0 * G_CGS * m_g / (r_cm * C_LIGHT_CGS**2)


def relativistic_grav_accel(m_g, r_cm, p_cgs, rho_gcm3) -> float:
    """Accelerazione gravitazionale radiale con correzione relativistica
    approssimata "Case A" (Marek et al. 2006), formula di modulo sopra
    esattamente come scritta.

    CORREZIONE (da review, vedi STATUS.md piano Step 5): MAI solleva
    un'eccezione qui, anche per stati numericamente estremi (r_cm<=0,
    m_g<=0, rho_gcm3<=0, compattezza>=1) — questa funzione e' chiamata da
    `_shell_dvdt`, sul path del RHS passato a solve_ivp, che non cattura
    eccezioni sollevate al suo interno (vedi nota di modulo). Usa invece un
    CLAMP numerico sulla compattezza (COMPACTNESS_CLAMP_MAX=0.99) prima di
    valutare il denominatore [1 - 2*G*m/(r*c^2)], stesso principio di
    V_FLOOR_CM3 in collasso.dynamics: solo una rete di sicurezza numerica
    residua, mai un'eccezione. L'unico meccanismo con cui l'esito
    "compattezza troppo alta" viene segnalato all'esterno resta l'evento
    terminale `_schwarzschild_proximity_event` in collasso.dynamics.

    Nota: rho_gcm3 e' la densita' della shell gia' disponibile nel
    chiamante (_shell_dvdt) — non viene ricalcolata qui, va passata come
    parametro.

    Nota di vettorizzazione: la formula e' puramente locale/pointwise per
    shell (nessuna dipendenza incrociata fra shell), quindi accetta sia
    scalari float sia np.ndarray per tutti e 4 i parametri (stessa
    convenzione delle funzioni vettorizzate di collasso.eos) — usata da
    _shell_dvdt passando direttamente gli array di TUTTE le shell, senza
    loop Python esplicito.
    """
    # Compattezza clampata (rete di sicurezza numerica, mai un'eccezione).
    # r_cm/m_g/rho_gcm3 non validati qui per lo stesso motivo: qualunque
    # guardia con ValueError romperebbe il RHS di solve_ivp. np.minimum (non
    # il builtin min()) per restare vettorizzata: questa funzione viene
    # chiamata da _shell_dvdt (collasso.dynamics) con m_g/r_cm/p_cgs/rho_gcm3
    # come array di TUTTE le shell contemporaneamente (formula puramente
    # locale/pointwise per shell, nessuna dipendenza incrociata) — min()
    # python solleverebbe ValueError ("ambiguous truth value") su array con
    # piu' di un elemento.
    compactness_raw = 2.0 * G_CGS * m_g / (r_cm * C_LIGHT_CGS**2)
    compactness_clamped = np.minimum(compactness_raw, COMPACTNESS_CLAMP_MAX)

    massa_efficace = m_g + 4.0 * np.pi * r_cm**3 * p_cgs / C_LIGHT_CGS**2
    fattore_pressione = 1.0 + p_cgs / (rho_gcm3 * C_LIGHT_CGS**2)
    denominatore = r_cm**2 * (1.0 - compactness_clamped)

    return -G_CGS * massa_efficace * fattore_pressione / denominatore
