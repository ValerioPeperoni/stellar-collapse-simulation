"""
collasso.dynamics — dinamica shell Lagrangiane del collasso gravitazionale
(Step 4).

Schema numerico: N shell di massa fissa (Lagrangiane, Delta_m_i costante nel
tempo), griglia sfalsata in stile von Neumann-Richtmyer, stato = raggi di
bordo r_i(t) e velocita' v_i(t)=dr_i/dt, r_0=0 (centro) sempre implicito, mai
una variabile di stato. Gravita' Newtoniana + gradiente di pressione
dall'EOS ESATTA di Chandrasekhar (`collasso.eos.pressure_chandrasekhar`,
Step 3) — mai `pressure_non_relativistic`/`pressure_ultrarelativistic` come
forza motrice della dinamica (questi ultimi servono solo per costruire
l'equilibrio di test, vedi `polytrope_equilibrium_rho_c_gcm3`).

Vincoli di modello per questo step (da CLAUDE.md e piano Step 4, non
negoziabili):
- Gravita' SOLO Newtoniana: nessuna correzione relativistica sul nucleo
  (responsabilita' esclusiva dello Step 5).
- Nessun limite TOV, nessuna classificazione del remnant (Step 6).
- Nessun termine di raffreddamento neutrini (limite di modello dichiarato).
- Nessuna rotazione/campo magnetico (limiti gia' esclusi a monte).
- Nessuna viscosita' artificiale: limite di modello dichiarato
  esplicitamente (possibile "ringing" numerico in forte compressione),
  accettabile perche' questo step si ferma all'innesco del collasso (soglia
  centrale o shell crossing), non al bounce.

Nota REV. 2 (correzione bloccante 2 del reviewer): il sintomo estremo del
limite "nessuna viscosita' artificiale" — l'attraversamento fra shell
consecutive — NON viene piu' segnalato sollevando un'eccezione dentro il RHS
passato a `solve_ivp` (scipy non la catturerebbe: crash non gestito del
processo). E' invece gestito con un secondo evento terminale pulito
(`_shell_crossing_event`), accompagnato da un floor numerico fisso
(`V_FLOOR_CM3`) applicato al volume di shell dentro il RHS come rete di
sicurezza residua (mai un'eccezione).

Nota di performance (obbligatoria, vedi piano 1.7): `_shell_dvdt` deve
restare vettorizzata (nessun loop Python esplicito su range(n)) — con
`N_SHELLS_DEMO_DEFAULT=200` e `T_MAX_FREE_FALL_MULTIPLIER=10`, un'implementazione
a loop esplicito rischierebbe tempi di esecuzione eccessivi nello script
demo.

Nota REV. 3 (limite strutturale del bordo libero ESTERNO — dichiarato come
limite di modello, stesso stile delle dichiarazioni sopra, vedi anche il
docstring di `_shell_dvdt` sotto per il dettaglio implementativo): la
discretizzazione "densita' media di shell (delta_m/V) + differenza in avanti
verso P=0 al bordo libero" (termine `press_ext` in `_shell_dvdt`) NON
converge a zero sull'ultima shell per un politropo con densita' nulla in
superficie (comportamento a legge di potenza, rho ~ s^n vicino al bordo,
s = distanza dalla superficie). L'errore converge invece a un valore limite
non nullo, universale nella forma `|1 - 2/(n+1)^((n+1)/n)|` (derivato dal
reviewer, confermato dal planner; per n=3/2 vale ~=0.5656 — vedi STATUS.md,
piano Step 4 REV. 3). E' un limite STRUTTURALE dello schema (non risolvibile
aumentando N) ma confinato a un numero FISSO e piccolo di shell vicino al
bordo (frazione di massa trascurabile, O(1/N) per shell) — non influenza la
dinamica macroscopica del collasso, che dipende dalle shell interne (dove lo
schema converge correttamente) e dalla massa totale/gravita', non dal
dettaglio locale di pressione all'ultima shell.

Nota REV. 4 (secondo limite strutturale, al bordo INTERNO/centro — stessa
natura qualitativa del limite REV. 3 sopra, ma un fenomeno DISTINTO, mai
diagnosticato dal reviewer/planner prima dell'indagine forense REV. 4, vedi
STATUS.md, "Piano Step 4 rivisto dopo indagine forense"): anche la shell
PIU' INTERNA (indice 0, adiacente al centro r=0, dove `press_int`/`grav`
usano `m_enclosed_g[0]=delta_m_g[0]` e `r_cm[0]`) mostra un rapporto
|accelerazione netta|/|accelerazione gravitazionale| che NON converge a zero
aumentando N, ma a un plateau finito non nullo (~0.056-0.058), confermato
numericamente su 4 ordini di grandezza di N (50->6400, convergenza liscia e
monotona: 0.05434, 0.05519, 0.05536, 0.05560, ...). Come per il bordo
esterno, e' un limite STRUTTURALE della discretizzazione (non di precisione
numerica: i valori assoluti coinvolti sono lontani da underflow) — verificato
essere un effetto della discretizzazione e non della fisica sostituendo
`pressure_chandrasekhar` con la stessa `pressure_non_relativistic` usata per
costruire l'equilibrio di test (il plateau resta, ~0.057). Anche questo
limite e' confinato a un numero FISSO e piccolo di shell (la sola shell
centrale, frazione di massa O(1/N)) e non influenza la dinamica
macroscopica del collasso per lo stesso motivo del bordo esterno. Entrambi i
limiti (bordo esterno REV. 3 e bordo interno/centro REV. 4) sono verificati
quantitativamente in `tests/test_dynamics.py`
(`test_polytrope_equilibrium_initial_acceleration_convergence`), che dalla
REV. 4 usa una metrica aggregata pesata in massa su tutte le shell (invece
di escludere manualmente le shell di bordo per indice, come nella REV. 3) —
vedi il docstring di quel test per i valori empirici osservati e per una
scoperta correlata (la metrica aggregata risulta in pratica dominata dal
plateau del bordo esterno, non dal bulk interno come inizialmente assunto
dal piano REV. 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

from collasso.constants import C_LIGHT_CGS, G_CGS
from collasso.eos import pressure_chandrasekhar
from collasso.relativistic import COMPACTNESS_SAFETY_LIMIT, relativistic_grav_accel

# --- Nota Step 5 (correzioni relativistiche sul nucleo, vedi STATUS.md,
#     piano Step 5): questo modulo e' esteso, non riscritto. Il termine
#     sorgente di gravita' in _shell_dvdt puo' ora essere sostituito da
#     `relativistic_grav_accel` (collasso.relativistic) quando
#     `relativistic=True` viene propagato da simulate_collapse; il resto
#     dello schema (pressione, shell Lagrangiane, eventi di soglia/shell
#     crossing gia' esistenti) resta identico. Vedi anche il nuovo evento
#     _schwarzschild_proximity_event sotto.

# --- Costanti e parametri numerici fissati dal planner (non a discrezione
#     del coder), vedi STATUS.md, piano Step 4 REV. 2 ------------------------

R_MIN_FRAC_DEFAULT = 0.1
N_POINTS_DEFAULT = 500

# t_max_s = T_MAX_FREE_FALL_MULTIPLIER * free_fall_time_s(rho_c_iniziale);
# simulate_collapse NON calcola t_max_s da sola (parametro obbligatorio,
# nessun default nella firma), per tenere esplicito il legame fra la scala
# temporale scelta e la densita' centrale iniziale della specifica
# configurazione.
#
# Valore alzato da 10.0 a 15.0 nel ciclo "Integrazione catalogo reale"
# (successivo a Step 8): con le masse REALI del nucleo (O'Connor & Ott
# 2011) al posto dei placeholder, s20 e' il caso piu' marginalmente
# supra-Chandrasekhar (massa_nucleo/M_Ch=102.3%) e il profilo
# auto-consistente (n_politropico~2.976, derivato da n_eff_chandrasekhar)
# richiede empiricamente ~12.6x t_ff (Newtoniano) / ~11x t_ff (con
# correzione relativistica) per raggiungere la soglia di collasso, contro
# 10x per s15/s25. 15.0 verificato con margine su tutti e tre i
# progenitori del catalogo di riferimento, in entrambe le modalita'
# relativistic=True/False.
T_MAX_FREE_FALL_MULTIPLIER = 15.0

# Floor numerico positivo fisso (cm^3) applicato al volume di ciascuna shell
# dentro _shell_dvdt, per evitare NaN/divisioni per zero SENZA mai sollevare
# un'eccezione dentro il RHS passato a solve_ivp (vedi nota di modulo sopra).
# Non e' il meccanismo che rileva/segnala lo shell crossing come esito
# (quello e' l'evento terminale _shell_crossing_event) — solo una rete di
# sicurezza numerica residua.
V_FLOOR_CM3 = 1e-30


@dataclass
class CollapseSolution:
    """Risultato della simulazione della dinamica shell Lagrangiane.

    Convenzione di shape (dichiarata esplicitamente per evitare bug di
    trasposizione a valle, es. Step 7 animazione): righe = shell, colonne =
    tempo — stessa convenzione di `solve_ivp.y`, NESSUNA trasposizione.

    Campi:
        t_s: array dei tempi campionati (lunghezza T, secondi).
        r_cm: raggi di bordo shell, shape (N, T); riga i = traiettoria
            temporale della shell i.
        v_cms: velocita' di bordo shell, shape (N, T), stessa convenzione.
        rho_c_gcm3_t: densita' della shell 1 (la piu' interna) nel tempo
            (lunghezza T): delta_m_g[0] / ((4/3)*pi*r_cm[0,:]**3).
        collapsed: True se uno dei due eventi terminali (soglia centrale o
            shell crossing) e' scattato prima di t_max_s.
        t_collapse_s: tempo (s) dell'evento che ha terminato l'integrazione,
            o None se `collapsed` e' False.
        collapse_reason: meccanismo che ha determinato la terminazione.
            Valori ammessi: "r_min_threshold" (soglia centrale raggiunta),
            "shell_crossing" (shell collassate/incrociate),
            "near_schwarzschild" (Step 5, solo se relativistic=True:
            compattezza massima su tutte le shell >= COMPACTNESS_SAFETY_LIMIT),
            None se `collapsed` e' False.

    Nota: se un evento terminale scatta prima di t_max_s, scipy.integrate.
    solve_ivp restringe automaticamente t_eval/sol.t/sol.y ai soli punti
    antecedenti l'evento — comportamento nativo di scipy, non richiede
    gestione manuale aggiuntiva qui, ma va tenuto presente da chi consuma
    questo oggetto (gli array possono avere meno di n_points colonne).
    """

    t_s: np.ndarray
    r_cm: np.ndarray
    v_cms: np.ndarray
    rho_c_gcm3_t: np.ndarray
    collapsed: bool
    t_collapse_s: float | None
    collapse_reason: str | None


def free_fall_time_s(rho_gcm3):
    """Tempo di caduta libera per una sfera uniforme di densita' rho_gcm3.

    t_ff = sqrt(3*pi/(32*G_CGS*rho_gcm3)) (formula standard). Vettorizzata
    (numpy). Guardia: rho_gcm3 > 0 stretto, altrimenti ValueError (stesso
    stile di collasso.eos._validate_rho).
    """
    rho_arr = np.asarray(rho_gcm3, dtype=float)
    if np.any(rho_arr <= 0):
        raise ValueError(
            f"rho_gcm3 deve essere strettamente positivo (>0); trovato "
            f"valore minimo={np.min(rho_arr)}"
        )
    t_ff = np.sqrt(3.0 * np.pi / (32.0 * G_CGS * rho_arr))
    if np.isscalar(rho_gcm3) or (isinstance(rho_gcm3, np.ndarray) and rho_gcm3.ndim == 0):
        return float(t_ff)
    return t_ff


def build_initial_shells(r_cm, rho_gcm3, n_shells: int):
    """Costruisce N shell Lagrangiane equispaziate in massa a partire da un
    profilo fisico continuo (r_cm, rho_gcm3), tipicamente da
    `collasso.lane_emden.physical_profile`.

    Algoritmo (fissato dal planner, non a discrezione del coder):
    - integra la massa cumulativa M(r) per quadratura trapezoidale;
    - sceglie N shell equispaziate in massa da M_total/N a M_total;
    - ricava i raggi di bordo per interpolazione inversa di M(r).

    Ritorna:
        (delta_m_g, m_enclosed_g, r0_cm): array di lunghezza n_shells.

    Nota: r_cm[0] non e' esattamente 0 per costruzione di `physical_profile`
    (xi0=1e-4), quindi si trascura una quantita' di massa trascurabile fra
    r=0 e r_cm[0] — stessa approssimazione gia' accettata da Lane-Emden
    (Step 2), non un nuovo limite introdotto qui.
    """
    if n_shells < 2:
        raise ValueError(f"n_shells={n_shells} deve essere >= 2.")

    r_cm = np.asarray(r_cm, dtype=float)
    rho_gcm3 = np.asarray(rho_gcm3, dtype=float)

    if len(r_cm) != len(rho_gcm3):
        raise ValueError(
            f"r_cm (len={len(r_cm)}) e rho_gcm3 (len={len(rho_gcm3)}) "
            "devono avere la stessa lunghezza."
        )
    # Nota implementativa (deviazione tecnica minima dal testo letterale del
    # piano, non una modifica fisica/di tolleranza): il piano richiede
    # "rho_gcm3 tutto >0", ma il campo `rho_gcm3` restituito da
    # `lane_emden.physical_profile` puo' legittimamente valere ESATTAMENTE
    # 0.0 all'ultimo punto di griglia (xi=xi1, la superficie della stella,
    # dove la densita' fisica si annulla) — dipende dal segno del residuo
    # numerico di theta a xi1 (vedi docstring di `LaneEmdenSolution.theta`,
    # Step 2): un residuo negativo viene clippato a 0 da `np.maximum`,
    # dando rho=0 esatto; un residuo positivo da' invece un valore
    # piccolissimo ma strettamente positivo. Verificato empiricamente: al
    # piu' UN solo punto di coda (mai piu' di uno) puo' annullarsi in
    # questo modo. Si richiede quindi rho_gcm3 >= 0 (non-negativo, non piu'
    # strettamente positivo), che accetta questo caso di bordo fisicamente
    # corretto (densita' nulla alla superficie) senza indebolire il
    # controllo su valori negativi (comunque non fisici, ancora rifiutati).
    if np.any(rho_gcm3 < 0):
        raise ValueError("rho_gcm3 deve essere non-negativo ovunque (trovato valore negativo).")
    if not np.all(np.diff(r_cm) > 0):
        raise ValueError("r_cm deve essere strettamente crescente.")

    m_r = cumulative_trapezoid(4.0 * np.pi * r_cm**2 * rho_gcm3, r_cm, initial=0.0)
    m_total = m_r[-1]

    m_targets = np.linspace(m_total / n_shells, m_total, n_shells)
    r0_cm = np.interp(m_targets, m_r, r_cm)

    delta_m_g = np.diff(np.concatenate(([0.0], m_targets)))
    m_enclosed_g = m_targets

    return delta_m_g, m_enclosed_g, r0_cm


def polytrope_equilibrium_rho_c_gcm3(
    k_nr_cgs: float,
    m_g: float,
    xi1: float,
    dtheta_dxi_at_xi1: float,
) -> float:
    """Densita' centrale di un politropo n=3/2 in equilibrio idrostatico
    Newtoniano autoconsistente, data la massa totale `m_g` e la costante
    K_NR (limite non-relativistico, P_NR=K_NR*rho^(5/3)).

    Formula chiusa verificata algebricamente dal planner (STATUS.md, piano
    Step 4 REV. 2, "Nota di verifica del planner"), da usare esattamente
    come scritta:
        geom_term = -xi1**2 * dtheta_dxi_at_xi1
        rho_c_gcm3 = ( m_g / (4*pi*geom_term*(2.5*k_nr_cgs/(4*pi*G_CGS))**1.5) )**2

    Nota: questa funzione e' specifica del caso n=3/2 (l'esponente 2.5=n+1 e
    la potenza 1.5 nella formula sono validi solo per n=3/2); NON
    generalizzare a n arbitrario senza consultare il planner.
    """
    geom_term = -(xi1**2) * dtheta_dxi_at_xi1
    if geom_term <= 0:
        raise ValueError(
            f"geom_term=-xi1^2*dtheta_dxi_at_xi1={geom_term} non positivo "
            "(condizione non fisica)."
        )
    if k_nr_cgs <= 0:
        raise ValueError(f"k_nr_cgs={k_nr_cgs} deve essere strettamente positivo.")
    if m_g <= 0:
        raise ValueError(f"m_g={m_g} deve essere strettamente positivo.")

    rho_c_gcm3 = (
        m_g / (4.0 * np.pi * geom_term * (2.5 * k_nr_cgs / (4.0 * np.pi * G_CGS)) ** 1.5)
    ) ** 2
    return rho_c_gcm3


def _shell_dvdt(r_cm, m_enclosed_g, delta_m_g, ye, relativistic: bool = False) -> np.ndarray:
    """Accelerazioni radiali di TUTTE le shell (lunghezza N), vettorizzata
    (nessun loop Python esplicito sulle shell).

    Step 5 (relativistic=False di default): comportamento IDENTICO a Step 4
    quando relativistic=False (nessuna riga di logica gravitazionale
    cambiata sul path di default). Quando relativistic=True, il branch
    sotto sostituisce SOLO il termine sorgente di gravita' `grav` con
    `relativistic_grav_accel` (collasso.relativistic, potenziale effettivo
    "Case A" di Marek et al. 2006) — il termine di pressione
    (press_int/press_ext) NON viene toccato, limite dichiarato
    esplicitamente: questa non e' una correzione GR completa
    dell'idrodinamica, solo del termine sorgente gravitazionale.

    Algoritmo fissato dal planner (STATUS.md, piano Step 4 REV. 2, punto
    1.4): gravita' Newtoniana + gradiente di pressione dall'EOS esatta di
    Chandrasekhar, floor numerico V_FLOOR_CM3 sul volume di shell (MAI
    un'eccezione — vedi nota di modulo).

    Nota REV. 3 (limite strutturale del bordo libero ESTERNO, dichiarato
    anche nel docstring di modulo sopra): `rho = delta_m_g / v` e' una
    densita' MEDIA di volume di shell, non la densita' puntuale al bordo.
    Per l'ultima shell, combinata con `press_ext` (differenza in avanti
    verso P=0 al bordo libero), questo introduce un errore sistematico che
    NON si annulla aumentando il numero di shell quando il profilo ha
    densita' nulla in superficie (rho ~ s^n, s = distanza dalla superficie):
    il rapporto |accelerazione netta|/|accelerazione gravitazionale|
    sull'ultima shell converge a un valore limite universale non nullo
    `|1 - 2/(n+1)^((n+1)/n)|` (per n=3/2, ~=0.5656), non a zero. E' un
    limite strutturale dello schema al bordo libero, confinato a un numero
    fisso e piccolo di shell (massa trascurabile), non un errore di
    implementazione — vedi STATUS.md, piano Step 4 REV. 3.

    Nota REV. 4 (secondo limite strutturale, al bordo INTERNO/centro, vedi
    docstring di modulo per il dettaglio completo): la stessa natura
    qualitativa di errore sistematico si presenta anche alla shell 0 (la
    piu' interna, adiacente al centro implicito r=0), dove `grav[0]` usa
    `m_enclosed_g[0]=delta_m_g[0]` (la massa dell'intera prima shell, non
    una densita' puntuale al centro) e `press_int[0]` usa la densita' media
    di volume delle shell 0 e 1. Il rapporto |accelerazione netta|/
    |accelerazione gravitazionale| alla shell 0 converge, aumentando N, a un
    plateau finito non nullo (~0.056-0.058), non a zero — confermato non
    essere un problema di precisione numerica (valori lontani da underflow)
    ne' un mismatch di EOS (persiste sostituendo pressure_chandrasekhar con
    pressure_non_relativistic). Fenomeno DISTINTO dal limite REV. 3 (bordo
    esterno), mai diagnosticato prima dell'indagine forense REV. 4 — vedi
    STATUS.md, piano Step 4 REV. 4, e il test
    `test_polytrope_equilibrium_initial_acceleration_convergence`.
    """
    r_cm = np.asarray(r_cm, dtype=float)
    m_enclosed_g = np.asarray(m_enclosed_g, dtype=float)
    delta_m_g = np.asarray(delta_m_g, dtype=float)

    r_full = np.concatenate(([0.0], r_cm))
    v_raw = (4.0 / 3.0) * np.pi * (r_full[1:] ** 3 - r_full[:-1] ** 3)
    v = np.maximum(v_raw, V_FLOOR_CM3)

    rho = delta_m_g / v
    p = pressure_chandrasekhar(rho, ye)

    if relativistic:
        # Formula puramente locale/pointwise per shell (nessuna dipendenza
        # incrociata fra shell) -> vettorizzata direttamente sugli array
        # m_enclosed_g/r_cm/p/rho, senza loop Python esplicito (vedi Nota
        # di performance nel docstring di modulo). Nota di segno: la
        # formula "Case A" (collasso.relativistic.relativistic_grav_accel)
        # include gia' il segno negativo (attrattivo), esattamente come
        # -G_CGS*m_enclosed_g/r_cm**2 del ramo Newtoniano sotto — nessuna
        # negazione aggiuntiva qui (verificato contro il limite di campo
        # debole, test 5.1/5.2 di STATUS.md piano Step 5).
        grav = relativistic_grav_accel(m_enclosed_g, r_cm, p, rho)
    else:
        grav = -G_CGS * m_enclosed_g / r_cm**2

    press_int = (
        -4.0 * np.pi * r_cm[:-1] ** 2 * (p[1:] - p[:-1]) / ((delta_m_g[:-1] + delta_m_g[1:]) / 2.0)
    )
    press_ext = 4.0 * np.pi * r_cm[-1] ** 2 * p[-1] / (delta_m_g[-1] / 2.0)

    dvdt = grav.copy()
    dvdt[:-1] += press_int
    dvdt[-1] += press_ext
    return dvdt


def _collapse_rhs(t, y, m_enclosed_g, delta_m_g, ye, n, r_min_cm, relativistic=False) -> np.ndarray:
    """RHS del sistema del primo ordine (r_1..r_N, v_1..v_N) passato a
    solve_ivp. Non solleva mai eccezioni per condizioni fisiche/numeriche
    (solo _shell_dvdt, che usa il floor V_FLOOR_CM3, e ora anche il clamp
    COMPACTNESS_CLAMP_MAX dentro relativistic_grav_accel) — coerente con la
    correzione bloccante 2 del reviewer (REV. 2) e con la correzione
    equivalente di Step 5 per la gravita' relativistica.

    Nota implementativa (deviazione tecnica minima dal testo letterale del
    piano, non fisica/numerica): il piano descriveva `args=(r_min_cm,)`
    dedicato al solo evento di soglia e `args=(n,)` dedicato al solo evento
    di shell crossing, distinti dall'args=(m_enclosed_g, delta_m_g, ye, n)
    di `fun`. In pratica `scipy.integrate.solve_ivp` passa UNA SOLA tupla
    `args` comune a `fun` e a TUTTI gli `events` (non e' possibile avere
    args diversi per funzioni diverse nella stessa chiamata) — vincolo
    dell'API di scipy non esplicitato nel piano. Risolto qui accorpando
    tutti i parametri extra necessari (inclusi n e r_min_cm) in un'UNICA
    tupla comune `(m_enclosed_g, delta_m_g, ye, n, r_min_cm)`, passata a
    `fun` e a entrambi gli eventi; ciascuna funzione ignora i parametri che
    non le servono. Nessuna formula fisica ne' tolleranza e' stata
    modificata per questo aggiustamento, solo la firma delle funzioni.

    Nota Step 5: la tupla comune e' stata allargata con `relativistic`
    (stesso principio della nota sopra, non riscoperto ma riapplicato) per
    poter propagare il flag anche al nuovo evento
    `_schwarzschild_proximity_event`, oltre che a `_shell_dvdt`.
    `relativistic` ha un default (`False`) SOLO per compatibilita' con
    chiamate dirette pre-esistenti che costruiscono la tupla args a 5
    elementi (Step 4, `tests/test_dynamics.py::test_shell_crossing_event_and_v_floor`,
    che chiama `_shell_crossing_event` senza passare `relativistic`) — non
    e' mai omesso da `simulate_collapse`, che lo passa sempre esplicitamente
    nella tupla comune passata a solve_ivp.
    """
    r = y[:n]
    v = y[n:]
    return np.concatenate([v, _shell_dvdt(r, m_enclosed_g, delta_m_g, ye, relativistic)])


def _collapse_threshold_event(t, y, m_enclosed_g, delta_m_g, ye, n, r_min_cm, relativistic=False) -> float:
    """Evento terminale: la shell piu' interna scende sotto r_min_cm.

    Stesso pattern di _theta_zero_event (Step 2): direction=-1, terminal=True.
    Firma allineata a quella comune di `_collapse_rhs` (vedi nota li');
    `relativistic` non e' usato da questa funzione, presente solo per
    coerenza di firma con la tupla args comune di solve_ivp.
    """
    return y[0] - r_min_cm


_collapse_threshold_event.direction = -1.0
_collapse_threshold_event.terminal = True


def _shell_crossing_event(t, y, m_enclosed_g, delta_m_g, ye, n, r_min_cm, relativistic=False) -> float:
    """Evento terminale (REV. 2, correzione bloccante 2): rileva
    l'attraversamento di shell (raggi di bordo consecutivi che si
    incrociano o collassano a spessore nullo, incluso il centro).

    Ritorna il gap minimo fra bordi consecutivi (incluso r_0=0). Quando
    questo gap si azzera/diventa negativo, l'evento scatta (direction=-1,
    terminal=True), fermando l'integrazione in modo pulito PRIMA che sia
    necessario ricorrere al floor V_FLOOR_CM3 dentro _shell_dvdt.

    Trattato come collapsed=True con collapse_reason="shell_crossing" in
    CollapseSolution, distinto da "r_min_threshold" — sintomo diretto del
    limite dichiarato "nessuna viscosita' artificiale" (possibile ringing
    numerico in forte compressione), non un errore di programmazione.

    Firma allineata a quella comune di `_collapse_rhs` (vedi nota li');
    `relativistic` non e' usato da questa funzione, presente solo per
    coerenza di firma con la tupla args comune di solve_ivp.
    """
    r_full = np.concatenate(([0.0], y[:n]))
    gaps = np.diff(r_full)
    return np.min(gaps)


_shell_crossing_event.direction = -1.0
_shell_crossing_event.terminal = True


def _schwarzschild_proximity_event(t, y, m_enclosed_g, delta_m_g, ye, n, r_min_cm, relativistic=False) -> float:
    """Evento terminale (Step 5, solo quando relativistic=True): rileva
    l'avvicinamento di QUALUNQUE shell al raggio di Schwarzschild.

    CORREZIONE (da review, STATUS.md piano Step 5): calcola la compattezza
    di TUTTE le shell (vettorizzato, m_enclosed_g/y[:n] sono array di
    lunghezza n), non solo della shell 0 — il piano REV. 1 assumeva senza
    dimostrazione che la shell piu' interna avesse sempre la compattezza
    massima, ipotesi non garantita in generale (m_enclosed_g cresce con
    l'indice ma r_cm cresce anch'esso, quindi il rapporto m/r non e'
    necessariamente monotono in indice di shell).

    Ritorna COMPACTNESS_SAFETY_LIMIT - max(compattezza su tutte le shell)
    (direction=-1, terminal=True): l'evento scatta quando la compattezza
    massima raggiunge/supera COMPACTNESS_SAFETY_LIMIT=0.9.

    Firma allineata a quella comune di `_collapse_rhs` (vedi nota li').
    """
    r_cm = y[:n]
    compactness_all = 2.0 * G_CGS * m_enclosed_g / (r_cm * C_LIGHT_CGS**2)
    return COMPACTNESS_SAFETY_LIMIT - np.max(compactness_all)


_schwarzschild_proximity_event.direction = -1.0
_schwarzschild_proximity_event.terminal = True


def simulate_collapse(
    delta_m_g,
    r0_cm,
    ye: float,
    t_max_s: float,
    r_min_frac: float = R_MIN_FRAC_DEFAULT,
    n_points: int = N_POINTS_DEFAULT,
    relativistic: bool = False,
) -> CollapseSolution:
    """Simula la dinamica shell Lagrangiane del collasso, a partire dalla
    condizione iniziale (delta_m_g, r0_cm) con shell inizialmente a riposo.

    Nota di modello: la condizione iniziale NON e' esattamente in equilibrio
    idrostatico rispetto alla EOS reale (mismatch gia' noto dal catalogo
    placeholder, Step 1/2) — non e' un bug, e' documentato qui.

    Vedi STATUS.md, piano Step 4 REV. 2, punto 1.6, per i dettagli
    dell'algoritmo (eventi terminali, max_step, determinazione di
    collapsed/t_collapse_s/collapse_reason da sol.t_events).

    Step 5: se relativistic=True, propaga il flag a `_shell_dvdt` (termine
    sorgente di gravita' sostituito con `relativistic_grav_accel`) e
    aggiunge un TERZO evento terminale `_schwarzschild_proximity_event`
    (compattezza massima su tutte le shell, vedi docstring dell'evento). Se
    relativistic=False (default), comportamento IDENTICO a Step 4: solo i
    primi due eventi, nessuna riga di logica gravitazionale cambiata sul
    path di default (verificato dal test di regressione 5.4, STATUS.md
    piano Step 5).
    """
    delta_m_g = np.asarray(delta_m_g, dtype=float)
    r0_cm = np.asarray(r0_cm, dtype=float)

    if t_max_s <= 0:
        raise ValueError(f"t_max_s={t_max_s} deve essere strettamente positivo.")
    if not (0.0 < r_min_frac < 1.0):
        raise ValueError(f"r_min_frac={r_min_frac} deve essere in (0, 1).")
    if len(delta_m_g) != len(r0_cm):
        raise ValueError(
            f"delta_m_g (len={len(delta_m_g)}) e r0_cm (len={len(r0_cm)}) "
            "devono avere la stessa lunghezza."
        )
    if len(r0_cm) < 2:
        raise ValueError(f"servono almeno 2 shell, trovate {len(r0_cm)}.")
    if np.any(delta_m_g <= 0):
        raise ValueError("delta_m_g deve essere strettamente positivo ovunque.")
    if not np.all(np.diff(r0_cm) > 0):
        raise ValueError("r0_cm deve essere strettamente crescente.")

    n = len(r0_cm)
    m_enclosed_g = np.cumsum(delta_m_g)
    y0 = np.concatenate([r0_cm, np.zeros(n)])
    r_min_cm = r_min_frac * r0_cm[0]

    max_step = t_max_s / 2000.0
    t_eval = np.linspace(0.0, t_max_s, n_points)

    # Step 5: il terzo evento (prossimita' al raggio di Schwarzschild) viene
    # costruito SOLO se relativistic=True — se relativistic=False, la lista
    # eventi resta identica a Step 4 (solo i primi due), come richiesto dal
    # piano (2.2).
    events = [_collapse_threshold_event, _shell_crossing_event]
    if relativistic:
        events.append(_schwarzschild_proximity_event)

    sol = solve_ivp(
        fun=_collapse_rhs,
        t_span=(0.0, t_max_s),
        y0=y0,
        method="RK45",
        rtol=1e-6,
        atol=1e-2,
        max_step=max_step,
        t_eval=t_eval,
        events=events,
        # Tupla comune passata a fun e a TUTTI gli eventi (vincolo di API
        # di scipy.integrate.solve_ivp, vedi nota in _collapse_rhs):
        # include sia n (usato da fun, _shell_crossing_event e
        # _schwarzschild_proximity_event) sia r_min_cm (usato solo da
        # _collapse_threshold_event) sia relativistic (Step 5, usato da fun
        # e implicitamente rilevante per _schwarzschild_proximity_event,
        # anche se quest'ultima non lo legge direttamente).
        args=(m_enclosed_g, delta_m_g, ye, n, r_min_cm, relativistic),
    )

    # Check esplicito ed incondizionato su sol.success (stesso pattern
    # obbligatorio di Step 2), PRIMA di processare gli eventi o costruire
    # CollapseSolution.
    if not sol.success:
        raise RuntimeError(f"integrazione della dinamica del collasso fallita: {sol.message}")

    # Determinazione di collapsed/t_collapse_s/collapse_reason: raccoglie
    # tutti gli eventi effettivamente scattati (in ORDINE DI PRIORITA' per
    # eventuali parita' esatte di tempo: r_min_threshold, poi
    # shell_crossing, poi near_schwarzschild — Step 5 generalizza, senza
    # cambiarla, la stessa convenzione arbitraria ma deterministica gia'
    # usata in Step 4), poi sceglie il tempo minimo. Per relativistic=False
    # questo blocco e' logicamente equivalente al codice originale di
    # Step 4 (stesso ordine di priorita', stesso risultato bit-per-bit
    # sugli stessi due eventi) — verificato dal test di regressione 5.4.
    event_hits: list[tuple[float, str]] = []
    if len(sol.t_events[0]) > 0:
        event_hits.append((float(sol.t_events[0][0]), "r_min_threshold"))
    if len(sol.t_events[1]) > 0:
        event_hits.append((float(sol.t_events[1][0]), "shell_crossing"))
    if relativistic and len(sol.t_events[2]) > 0:
        event_hits.append((float(sol.t_events[2][0]), "near_schwarzschild"))

    collapsed = len(event_hits) > 0
    if not collapsed:
        t_collapse_s = None
        collapse_reason = None
    else:
        # sort() e' stabile: a parita' esatta di tempo, l'ordine relativo
        # di inserimento sopra (= ordine di priorita' voluto) e' preservato.
        event_hits.sort(key=lambda item: item[0])
        t_collapse_s, collapse_reason = event_hits[0]

    r_cm = sol.y[:n, :]
    v_cms = sol.y[n:, :]
    t_s = sol.t
    rho_c_gcm3_t = delta_m_g[0] / ((4.0 / 3.0) * np.pi * r_cm[0, :] ** 3)

    return CollapseSolution(
        t_s=t_s,
        r_cm=r_cm,
        v_cms=v_cms,
        rho_c_gcm3_t=rho_c_gcm3_t,
        collapsed=collapsed,
        t_collapse_s=t_collapse_s,
        collapse_reason=collapse_reason,
    )
