"""
collasso.pipeline — orchestrazione pura della simulazione completa del
collasso gravitazionale stellare (Step 7).

`run_full_simulation(star_id)` compone, senza alcun I/O (nessun `print`,
nessuna scrittura su disco), tutte le funzioni gia' validate nei moduli
precedenti del progetto:
    - Step 1 (`collasso.catalog`): caricamento del progenitore scelto;
    - Step 2 (`collasso.lane_emden`): profilo di equilibrio iniziale;
    - Step 4/5 (`collasso.dynamics`): dinamica shell Lagrangiane, con
      correzione relativistica sul nucleo attivata ESPLICITAMENTE
      (`relativistic=True` passato alla chiamata, non un cambio del default
      di `simulate_collapse`, che resta `False`);
    - Step 6/Retrofit (`collasso.eos`, `collasso.tov`, `collasso.remnant`):
      massa di Chandrasekhar, limite di Oppenheimer-Volkoff per le EOS
      realistiche SLy e APR4 (mai il path storico/bacato
      `*_legacy_bug_rho_source`), classificazione del remnant.

Questo modulo aggiunge (Step 7) il bilancio energetico discreto cinetica +
autoenergia gravitazionale delle shell, calcolato da
`compute_collapse_energies` — vedi il docstring di quella funzione per le
formule esatte e i limiti dichiarati. La composizione include anche (ciclo
"Integrazione catalogo reale", successivo a Step 8) `collasso.eos.
chandrasekhar_k_deviation_pct`, formula gia' definita in `collasso.eos`
(non nuova QUI), usata per quantificare per ogni stella quanto il profilo
iniziale si discosti da un vero equilibrio idrostatico con la EOS di
Chandrasekhar — sempre atteso e non nullo per le stelle del catalogo di
riferimento, tutte supra-Chandrasekhar (vedi `SimulationResult.k_deviation_pct`).

Vincoli di modello per questo step (da CLAUDE.md, non negoziabili, invariati
rispetto agli step precedenti): nessuna nuova discretizzazione fisica viene
introdotta qui, solo composizione di funzioni gia' validate + un bilancio
energetico discreto di post-processing. Tutti gli 8 disclaimer fissi del
progetto (piu' un nono condizionale per le voci proxy del catalogo, vedi
`_build_disclaimers`) sono raccolti qui in `SimulationResult.disclaimers`
per essere testabili indipendentemente da come vengono poi stampati
dall'entry point (`scripts/run_simulation.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from collasso.catalog import Progenitor, get_progenitor_by_id, load_reference_catalog
from collasso.constants import G_CGS
from collasso.dynamics import (
    T_MAX_FREE_FALL_MULTIPLIER,
    CollapseSolution,
    build_initial_shells,
    free_fall_time_s,
    simulate_collapse,
)
from collasso.eos import chandrasekhar_k_deviation_pct, chandrasekhar_mass_msun
from collasso.eos_piecewise_polytrope import build_apr4, build_sly
from collasso.lane_emden import LaneEmdenSolution, PhysicalProfile, physical_profile, solve_lane_emden
from collasso.remnant import classify_remnant
from collasso.tov import RHO_C_GRID_REALISTIC_CGS, TOVSolution, find_ov_mass_limit

# Numero di shell Lagrangiane per la dinamica: stesso default gia' usato
# negli script demo precedenti (Step 4/5, N_SHELLS_DEMO_DEFAULT=200).
N_SHELLS = 200


@dataclass
class CollapseEnergies:
    """Bilancio energetico discreto (cinetica + autoenergia gravitazionale)
    calcolato all'istante iniziale e all'ULTIMO istante disponibile della
    simulazione (che coincide con un vero momento di collasso solo se
    `CollapseSolution.collapsed` e' True — vedi `compute_collapse_energies`).

    Campi:
        KE_finale_erg: energia cinetica totale delle shell all'ultimo
            istante disponibile.
        PE_iniziale_erg: autoenergia gravitazionale discretizzata delle
            shell a t=0 (negativa per definizione).
        PE_finale_erg: stessa quantita' all'ultimo istante disponibile.
        Delta_PE_erg: PE_finale_erg - PE_iniziale_erg (atteso negativo:
            energia liberata dal collasso).
        t_energie_s: istante temporale (s) a cui sono state valutate le
            quantita' "finali" sopra — coincide con
            `collapse_solution.t_s[-1]`, cioe' o il vero tempo di collasso
            (`t_collapse_s`, se `collapsed=True`) o l'ultimo tempo integrato
            prima di t_max_s (se `collapsed=False`, l'integrazione si e'
            semplicemente fermata senza un evento fisico).
    """

    KE_finale_erg: float
    PE_iniziale_erg: float
    PE_finale_erg: float
    Delta_PE_erg: float
    t_energie_s: float


def compute_collapse_energies(
    collapse_solution: CollapseSolution,
    delta_m_g: np.ndarray,
    m_enclosed_g: np.ndarray,
) -> CollapseEnergies:
    """Calcola il bilancio energetico discreto di una `CollapseSolution`.

    Energia cinetica finale (formula chiusa, nessuna nuova fisica):
        KE_finale_erg = sum(0.5*delta_m_g_i*v_cms_i(t_finale)**2)
    valutata all'ULTIMO istante disponibile (`collapse_solution.v_cms[:, -1]`)
    — sia che `collapsed` sia True (vero momento di collasso) sia che sia
    False (l'integrazione si e' fermata a t_max_s senza un evento fisico:
    in quel caso queste NON sono le energie di un vero momento di collasso,
    solo dell'ultimo stato integrato — vedi `CollapseEnergies.t_energie_s`
    e la gestione esplicita in `scripts/run_simulation.py`).

    Autoenergia gravitazionale discretizzata delle shell (formula
    CORRETTA dopo review, massa a META' shell invece che a bordo esterno,
    per ridurre il bias sistematico da somma di Riemann a estremo destro,
    O(1/N), sovrastima di |PE|, a un errore O(1/N^2) simile a una regola
    del punto medio):
        m_mid_i = m_enclosed_g_i - delta_m_g_i/2
        PE_erg(t) = -G_CGS * sum(m_mid_i * delta_m_g_i / r_cm_i(t))
    calcolata a t=0 (`r_cm[:, 0]`) e all'ultimo istante disponibile
    (`r_cm[:, -1]`); Delta_PE_erg = PE_finale - PE_iniziale (atteso
    negativo: energia liberata dal collasso).

    LIMITI DICHIARATI (da riportare sempre nei disclaimer a valle):
    - questa e' un'APPROSSIMAZIONE DISCRETA dell'autoenergia gravitazionale
      (somma di Riemann sulle shell), non validata contro un caso analitico
      a differenza di Lane-Emden/TOV altrove nel progetto — va dichiarata
      onestamente come tale, non presentata come precisa;
    - il bilancio energetico include SOLO cinetica + autoenergia
      gravitazionale, NON energia interna/termica ne' perdite per neutrini
      — `Delta_PE_erg + KE_finale_erg` NON e' una verifica di conservazione
      dell'energia (il bilancio non e' chiuso per costruzione, coerente con
      il limite "nessun raffreddamento neutrinico" gia' dichiarato dal
      progetto).
    """
    delta_m_g = np.asarray(delta_m_g, dtype=float)
    m_enclosed_g = np.asarray(m_enclosed_g, dtype=float)

    v_finale_cms = collapse_solution.v_cms[:, -1]
    KE_finale_erg = float(np.sum(0.5 * delta_m_g * v_finale_cms**2))

    m_mid_g = m_enclosed_g - delta_m_g / 2.0

    r_iniziale_cm = collapse_solution.r_cm[:, 0]
    r_finale_cm = collapse_solution.r_cm[:, -1]

    PE_iniziale_erg = float(-G_CGS * np.sum(m_mid_g * delta_m_g / r_iniziale_cm))
    PE_finale_erg = float(-G_CGS * np.sum(m_mid_g * delta_m_g / r_finale_cm))
    Delta_PE_erg = PE_finale_erg - PE_iniziale_erg

    t_energie_s = float(collapse_solution.t_s[-1])

    return CollapseEnergies(
        KE_finale_erg=KE_finale_erg,
        PE_iniziale_erg=PE_iniziale_erg,
        PE_finale_erg=PE_finale_erg,
        Delta_PE_erg=Delta_PE_erg,
        t_energie_s=t_energie_s,
    )


def _build_disclaimers(k_deviation_pct: float, nota_proxy: str = "") -> list[str]:
    """Costruisce la lista degli 8 disclaimer del progetto (testo libero,
    ma ciascuna voce deve contenere le parole chiave verificate dai test —
    vedi STATUS.md, piano Step 7, "Contenuto esatto di disclaimers"), piu'
    un NONO disclaimer condizionale se `nota_proxy` non e' vuota (ciclo
    "Espansione catalogo + Betelgeuse").

    `k_deviation_pct`: scarto percentuale per QUESTA stella fra la
    costante politropica implicita nel profilo iniziale e quella reale
    della EOS di Chandrasekhar (vedi `collasso.eos.
    chandrasekhar_k_deviation_pct`) — interpolato nel disclaimer 2 cosi'
    che il numero per-stella sia sempre visibile nell'output finale, non
    solo nella documentazione interna (richiesta esplicita dell'utente).

    `nota_proxy`: da `collasso.catalog.Progenitor.nota_proxy` — se non
    vuota, questa voce del catalogo non rappresenta un modello calcolato
    specificamente per l'oggetto nominato, ma un modello generico riusato
    come proxy dichiarato esplicitamente (es. "betelgeuse"). Quando non
    vuota, viene aggiunto un NONO disclaimer con questo testo, cosi'
    l'etichettatura "proxy" compare sia nel catalogo (`fonte`/`nota_proxy`)
    sia nell'output finale di ogni esecuzione, mai solo in un posto solo.
    """
    disclaimers = [
        # 1. Provenienza catalogo (aggiornato nel ciclo "Integrazione
        #    catalogo reale"/"Espansione catalogo", successivo a Step 8) —
        #    parola chiave: "O'Connor" (citazione di letteratura reale).
        "Catalogo dei progenitori: per le 6 stelle di riferimento "
        "(s15/s20/s25/s30/s35/s40, piu' la voce proxy 'betelgeuse'), massa "
        "del nucleo e ye sono dati di LETTERATURA reale (O'Connor & Ott "
        "2011, ApJ 730:70, Tabella 1, dati Woosley Heger & Weaver 2002) - "
        "solo 6 stelle, non un campionamento statistico del catalogo "
        "completo; non usare per conclusioni quantitative sulla "
        "popolazione di supernovae.",
        # 2. Densita' centrale/n/raggio derivati internamente (non
        #    letteratura) — parola chiave: "derivat".
        "Densita' centrale, indice politropico e raggio iniziale del "
        "nucleo NON sono dati di letteratura: la densita' centrale e' "
        "scelta come valore fisico plausibile, n politropico e raggio "
        "sono DERIVATI internamente (collasso.eos.n_eff_chandrasekhar + "
        "collasso.lane_emden) a partire dalla massa tabulata - inoltre "
        "questa stella ha massa del nucleo sopra la massa di "
        "Chandrasekhar, quindi NESSUN vero equilibrio idrostatico esiste "
        "per questa massa con EOS di Chandrasekhar esatta a nessuna "
        f"densita' centrale: il profilo implica una costante politropica "
        f"piu' 'rigida' del {abs(k_deviation_pct):.1f}% rispetto alla EOS "
        "esatta (scarto K, vedi collasso.eos.chandrasekhar_k_deviation_pct) "
        "- NON un errore di modellazione, ma la misura diretta e attesa di "
        "quanto questo nucleo sia supra-Chandrasekhar (quanto e' instabile). "
        "Il profilo va inteso come condizione iniziale della dinamica, non "
        "come equilibrio.",
        # 3. Nessun raffreddamento neutrinico (Step 4+) — parola chiave:
        #    "neutrini".
        "Nessun trasporto di neutrini: il progetto usa solo un termine di "
        "raffreddamento neutrini semplificato/assente (limite dichiarato "
        "in CLAUDE.md), non un trasporto radiativo completo - l'energia "
        "persa per neutrini non e' inclusa nel bilancio energetico.",
        # 4. Nessuna viscosita' artificiale (Step 4) — parola chiave:
        #    "viscosità" o "viscosita".
        "Nessuna viscosita' artificiale nello schema numerico delle shell "
        "Lagrangiane: possibile 'ringing' numerico in forte compressione, "
        "accettabile perche' la dinamica si ferma all'innesco del collasso "
        "(soglia centrale, shell crossing o prossimita' a Schwarzschild), "
        "non al bounce.",
        # 5. Correzione relativistica approssimata "Case A" (Step 5) —
        #    parola chiave: "Case A" o "approssimat" + "relativistic"/"GR".
        "Correzione relativistica sul nucleo: potenziale gravitazionale "
        "effettivo APPROSSIMATO 'Case A' (Marek, Dimmelmeier, Janka, "
        "Mueller & Buras 2006), non idrodinamica GR completa - solo il "
        "termine sorgente di gravita' e' corretto relativisticamente, il "
        "gradiente di pressione resta Newtoniano.",
        # 6. EOS SLy/APR4 fenomenologiche (Retrofit Step 6) — parola
        #    chiave: "fenomenolog".
        "EOS realistiche SLy e APR4 (Read et al. 2009): sono fit "
        "fenomenologici a politropi a tratti calibrati su calcoli "
        "nucleari, NON EOS a molti corpi complete - accurate entro ~1% "
        "sulle relazioni massa-raggio delle rispettive EOS originali.",
        # 7. Nessuna rotazione/campo magnetico — parola chiave: "rotazione".
        "Rotazione e campo magnetico completamente trascurati in tutto il "
        "progetto (limite di modello dichiarato in CLAUDE.md), nonostante "
        "possano influenzare significativamente la dinamica reale del "
        "collasso e la formazione del remnant.",
        # 8. Simmetria sferica (1D) — parola chiave: "simmetria sferica" o
        #    "1D".
        "Simmetria sferica (modello 1D): il collasso reale e' "
        "intrinsecamente tridimensionale (convezione, instabilita' SASI); "
        "questo modello a simmetria sferica non puo' catturare questi "
        "effetti, potenzialmente determinanti per l'esito reale "
        "(esplosione vs formazione diretta del remnant) - probabilmente il "
        "limite concettualmente piu' importante del progetto.",
    ]

    if nota_proxy:
        # 9. Solo per voci proxy (es. "betelgeuse") — parola chiave:
        #    "proxy". Ripete lo stesso testo del catalogo (nota_proxy) per
        #    non poter essere mancato leggendo solo i disclaimer.
        disclaimers.append(
            "Questa stella NON e' un modello calcolato specificamente per "
            f"l'oggetto nominato: {nota_proxy} - vedi il campo 'fonte' del "
            "catalogo (collasso.catalog) per la citazione completa della "
            "stima di massa reale usata per scegliere il modello proxy piu' "
            "vicino disponibile."
        )

    return disclaimers


@dataclass
class SimulationResult:
    """Risultato completo di `run_full_simulation`: tutti gli ingredienti
    numerici necessari sia all'output testuale strutturato
    (`scripts/run_simulation.py`) sia alla visualizzazione
    (`collasso.visualization`).

    Campi:
        star: progenitore scelto (`collasso.catalog.Progenitor`).
        lane_emden_solution: soluzione adimensionale di Lane-Emden.
        profile: profilo fisico iniziale (`collasso.lane_emden.PhysicalProfile`).
        delta_m_g: masse di shell Lagrangiane (fisse nel tempo), lunghezza
            N_SHELLS.
        m_enclosed_g: massa racchiusa cumulativa a bordo shell (fissa nel
            tempo), lunghezza N_SHELLS.
        t_ff_s: tempo di caduta libera dalla densita' centrale iniziale.
        t_max_s: tempo massimo di integrazione usato (T_MAX_FREE_FALL_MULTIPLIER*t_ff_s).
        collapse_solution: soluzione completa della dinamica (Step 4/5),
            con `relativistic=True` propagato ESPLICITAMENTE alla chiamata
            di `simulate_collapse` (il default della funzione resta False,
            non modificato).
        energies: bilancio energetico discreto (vedi `compute_collapse_energies`).
        m_chandrasekhar_msun: massa di Chandrasekhar Newtoniana classica
            (Step 3) per il `ye` della stella.
        m_tov_sly_msun / m_tov_apr4_msun: limite di Oppenheimer-Volkoff
            (massimo di massa) per le EOS realistiche SLy/APR4.
        rho_c_tov_sly_gcm3 / rho_c_tov_apr4_gcm3: densita' centrale a cui e'
            raggiunto il massimo di massa, per SLy/APR4.
        tov_sequence_sly / tov_sequence_apr4: sequenza completa di
            `TOVSolution` sulla griglia di densita' centrali (riusata dalla
            visualizzazione per il pannello massa-raggio, senza
            ricalcolare).
        remnant_class_sly / remnant_class_apr4: classificazione del
            remnant (`collasso.remnant.classify_remnant`) usando la soglia
            TOV rispettivamente di SLy/APR4.
        k_deviation_pct: scarto percentuale, per QUESTA stella, fra la
            costante politropica implicita nel profilo iniziale e quella
            reale della EOS di Chandrasekhar alla stessa densita'
            centrale/n (`collasso.eos.chandrasekhar_k_deviation_pct`) —
            misura diretta di quanto il nucleo sia supra-Chandrasekhar
            (nessun errore di modellazione, vedi docstring di quella
            funzione); interpolato anche nel disclaimer 2.
        disclaimers: lista degli 8 disclaimer fissi del progetto (9 per le
            voci proxy del catalogo, vedi `_build_disclaimers`), costruita
            qui per essere testabile
            indipendentemente dalla stampa a valle.
    """

    star: Progenitor
    lane_emden_solution: LaneEmdenSolution
    profile: PhysicalProfile
    delta_m_g: np.ndarray
    m_enclosed_g: np.ndarray
    t_ff_s: float
    t_max_s: float
    collapse_solution: CollapseSolution
    energies: CollapseEnergies
    m_chandrasekhar_msun: float
    m_tov_sly_msun: float
    m_tov_apr4_msun: float
    rho_c_tov_sly_gcm3: float
    rho_c_tov_apr4_gcm3: float
    k_deviation_pct: float
    tov_sequence_sly: list[TOVSolution]
    tov_sequence_apr4: list[TOVSolution]
    remnant_class_sly: str
    remnant_class_apr4: str
    disclaimers: list[str]


def run_full_simulation(star_id: str) -> SimulationResult:
    """Esegue la pipeline completa (equilibrio iniziale + dinamica +
    classificazione remnant + bilancio energetico) per il progenitore
    `star_id` del catalogo di riferimento. Nessun I/O (nessun print,
    nessuna scrittura su disco) — orchestrazione pura.

    Solleva:
        ValueError: se `star_id` non esiste nel catalogo (propagato da
            `get_progenitor_by_id`), o per condizioni non fisiche
            propagate dai moduli sottostanti.
    """
    # --- 1. Catalogo ---------------------------------------------------
    catalogo = load_reference_catalog()
    star = get_progenitor_by_id(catalogo, star_id)

    # --- 2. Profilo di equilibrio iniziale (Step 2) --------------------
    lane_emden_solution = solve_lane_emden(n=star.n_politropico)
    profile = physical_profile(lane_emden_solution, star.densita_centrale_gcm3, star.massa_nucleo_msun)

    # --- 3. Dinamica (Step 4/5) -----------------------------------------
    delta_m_g, m_enclosed_g, r0_cm = build_initial_shells(profile.r_cm, profile.rho_gcm3, N_SHELLS)

    t_ff_s = free_fall_time_s(star.densita_centrale_gcm3)
    t_max_s = T_MAX_FREE_FALL_MULTIPLIER * t_ff_s

    # relativistic=True passato ESPLICITAMENTE qui (non un cambio del
    # default di simulate_collapse, che resta False) — scelta dichiarata
    # nei disclaimer (voce 5), non nascosta.
    collapse_solution = simulate_collapse(
        delta_m_g, r0_cm, ye=star.ye, t_max_s=t_max_s, relativistic=True
    )

    # --- 4. Energie (formule chiuse, nessuna nuova fisica se non il
    #        bilancio discreto stesso) -----------------------------------
    energies = compute_collapse_energies(collapse_solution, delta_m_g, m_enclosed_g)

    # --- 5. Classificazione remnant (Step 6/Retrofit) --------------------
    m_chandrasekhar_msun = chandrasekhar_mass_msun(star.ye)

    rho_c_tov_sly_gcm3, m_tov_sly_msun, tov_sequence_sly = find_ov_mass_limit(
        RHO_C_GRID_REALISTIC_CGS, eos=build_sly()
    )
    rho_c_tov_apr4_gcm3, m_tov_apr4_msun, tov_sequence_apr4 = find_ov_mass_limit(
        RHO_C_GRID_REALISTIC_CGS, eos=build_apr4()
    )

    remnant_class_sly = classify_remnant(star.massa_nucleo_msun, m_chandrasekhar_msun, m_tov_sly_msun)
    remnant_class_apr4 = classify_remnant(star.massa_nucleo_msun, m_chandrasekhar_msun, m_tov_apr4_msun)

    # --- 5bis. Scarto K (quanto il nucleo e' supra-Chandrasekhar) --------
    k_deviation_pct = chandrasekhar_k_deviation_pct(
        star.densita_centrale_gcm3, star.ye, star.n_politropico, profile.alpha_cm
    )

    # --- 6. Disclaimer ----------------------------------------------------
    disclaimers = _build_disclaimers(k_deviation_pct, star.nota_proxy)

    return SimulationResult(
        star=star,
        lane_emden_solution=lane_emden_solution,
        profile=profile,
        delta_m_g=delta_m_g,
        m_enclosed_g=m_enclosed_g,
        t_ff_s=t_ff_s,
        t_max_s=t_max_s,
        collapse_solution=collapse_solution,
        energies=energies,
        m_chandrasekhar_msun=m_chandrasekhar_msun,
        m_tov_sly_msun=m_tov_sly_msun,
        m_tov_apr4_msun=m_tov_apr4_msun,
        rho_c_tov_sly_gcm3=rho_c_tov_sly_gcm3,
        rho_c_tov_apr4_gcm3=rho_c_tov_apr4_gcm3,
        k_deviation_pct=k_deviation_pct,
        tov_sequence_sly=tov_sequence_sly,
        tov_sequence_apr4=tov_sequence_apr4,
        remnant_class_sly=remnant_class_sly,
        remnant_class_apr4=remnant_class_apr4,
        disclaimers=disclaimers,
    )
