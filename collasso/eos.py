"""
collasso.eos — EOS del gas di elettroni completamente degenere e freddo
(T=0), formula esatta di Chandrasekhar (Step 3).

Riferimenti: Chandrasekhar 1939, *An Introduction to the Study of Stellar
Structure*; Shapiro & Teukolsky 1983, *Black Holes, White Dwarfs, and
Neutron Stars*, cap. 2-3.

Vincoli di modello per questo step (da CLAUDE.md e piano Step 3, non
negoziabili):
- Gas di elettroni completamente degenere e **freddo (T=0)**: nessun
  termine termico/entropico, nessuna dipendenza dalla temperatura.
- Questo modulo fornisce SOLO la funzione P(rho) (e Gamma1(rho), n_eff(rho),
  M_Ch(Ye)): NON reimposta l'equilibrio stellare (Lane-Emden, Step 2), NON
  introduce dinamica (Step 4), NON introduce correzioni relativistiche sul
  nucleo oltre alla EOS stessa (Step 5), NON introduce il limite TOV /
  classificazione remnant (Step 6).
- `chandrasekhar_mass_msun(ye)` e' la massa di Chandrasekhar **Newtoniana
  classica** per un politropo n=3 (indipendente da rho_c) — NON e' la
  soglia GR-corretta di instabilita' (quella resta esclusivamente il
  limite di Oppenheimer-Volkoff/TOV, Step 6).

Range di validazione di `ye` in questo modulo: (0, 1] — range fisico
GENERALE dell'EOS del gas di elettroni degenere, distinto dal range
catalogo-specifico 0.42-0.50 di `collasso.catalog` (quest'ultimo vincola
solo i progenitori del catalogo di riferimento, non la fisica generale
dell'EOS).
"""

from __future__ import annotations

import math

import numpy as np

from collasso.constants import (
    C_LIGHT_CGS,
    G_CGS,
    HBAR_CGS,
    M_ELECTRON_G,
    M_SUN_G,
    M_U_G,
)
from collasso.lane_emden import solve_lane_emden

# Soglia di transizione fra ramo a serie e ramo diretto di `chandrasekhar_f`
# (vedi docstring della funzione per la motivazione — cancellazione
# catastrofica, fissata dal planner dopo review numerica).
X_SERIES_THRESHOLD = 0.1

# Prefattore della pressione esatta di Chandrasekhar (erg/cm^3 per unita' di
# f(x)), calcolato una sola volta a tempo di import dalle costanti
# fondamentali cgs.
_P0_CGS = (M_ELECTRON_G**4 * C_LIGHT_CGS**5) / (24.0 * math.pi**2 * HBAR_CGS**3)

# Costante ricorrente nel calcolo della densita' numerica di elettroni.
_3_PI2 = 3.0 * math.pi**2


def _ensure_scalar_return(arr, original_input):
    """Ritorna arr come float scalare se original_input era scalare, altrimenti ndarray.

    Evita duplicazione del pattern np.isscalar / x.ndim == 0 ripetuto in
    chandrasekhar_f, _gamma1_of_x, _rho_from_x.
    """
    if np.isscalar(original_input) or (isinstance(original_input, np.ndarray) and original_input.ndim == 0):
        return float(arr)
    return arr


def _validate_rho(rho_gcm3):
    """Valida rho_gcm3 > 0 strettamente (scalare o elementwise su ndarray).

    Richiedere rho_gcm3 > 0 stretto evita naturalmente la forma
    indeterminata 0/0 di Gamma1 in x=0 (f(0)=0, Gamma1 non definita li'),
    senza bisogno di una guardia dedicata separata.
    """
    rho_arr = np.asarray(rho_gcm3, dtype=float)
    if np.any(rho_arr <= 0):
        raise ValueError(
            f"rho_gcm3 deve essere strettamente positivo (>0); trovato "
            f"valore minimo={np.min(rho_arr)}"
        )


def _validate_ye(ye):
    """Valida 0 < ye <= 1 (range fisico generale dell'EOS, non quello del
    catalogo, vedi docstring del modulo).
    """
    if not (0.0 < ye <= 1.0):
        raise ValueError(f"ye={ye} fuori dal range fisico atteso dell'EOS (0, 1].")


def fermi_x(rho_gcm3, ye):
    """Parametro di Fermi adimensionale x = p_F/(m_e*c).

    x = (hbar/(m_e*c)) * (3*pi^2*n_e)^(1/3), con n_e = ye*rho_gcm3/M_U_G.

    Vettorizzata: accetta sia scalari float sia np.ndarray.
    """
    _validate_rho(rho_gcm3)
    _validate_ye(ye)

    n_e = ye * np.asarray(rho_gcm3, dtype=float) / M_U_G
    x = (HBAR_CGS / (M_ELECTRON_G * C_LIGHT_CGS)) * (_3_PI2 * n_e) ** (1.0 / 3.0)
    return x


def chandrasekhar_f(x):
    """Building-block adimensionale f(x) della pressione esatta di
    Chandrasekhar, puro su x (nessuna dipendenza da rho/ye).

    Forma diretta esatta (Chandrasekhar 1939):
        f(x) = x*(2*x^2-3)*sqrt(1+x^2) + 3*asinh(x)

    REV. 2 — implementazione a DUE RAMI, fissata dal planner dopo review
    numerica per correggere un problema concreto di **cancellazione
    catastrofica**, NON un problema di overflow/underflow ne' di asinh in
    se': per x piccolo la formula diretta somma due termini di segno
    opposto e modulo quasi uguale (x*(2*x^2-3)*sqrt(1+x^2) ~ -3x,
    3*asinh(x) ~ +3x), il cui risultato atteso e' O(x^5) — la sottrazione
    fra numeri quasi uguali cancella molte cifre significative in
    floating point, con errore relativo sul valore calcolato che esplode
    ben oltre la precisione macchina utile. Il ramo a serie evita del
    tutto la sottrazione fra termini quasi opposti.

    - Per x < X_SERIES_THRESHOLD (0.1): sviluppo in serie troncato attorno
      a x=0 (3 termini, errore di troncamento relativo atteso ~1e-11 a
      x=X_SERIES_THRESHOLD, trascurabile rispetto a qualunque tolleranza
      di test di questo progetto):
          f_series(x) = (8/5)*x**5 - (4/7)*x**7 + (1/3)*x**9
    - Per x >= X_SERIES_THRESHOLD: formula diretta esatta (nessuna
      discontinuita' significativa al bordo, verificato dal test dedicato
      di continuita' del branching).

    Vettorizzata: il branching e' elementwise via np.where (entrambi i
    rami vengono calcolati sull'intero array e poi selezionati — accettato
    come leggermente ridondante ma coerente con il resto del modulo,
    preferibile a un loop Python esplicito).
    """
    x_arr = np.asarray(x, dtype=float)
    if np.any(x_arr < 0):
        raise ValueError(f"x deve essere >= 0; trovato valore minimo={np.min(x_arr)}")

    f_series = (8.0 / 5.0) * x_arr**5 - (4.0 / 7.0) * x_arr**7 + (1.0 / 3.0) * x_arr**9
    f_direct = x_arr * (2.0 * x_arr**2 - 3.0) * np.sqrt(1.0 + x_arr**2) + 3.0 * np.arcsinh(x_arr)

    f = np.where(x_arr < X_SERIES_THRESHOLD, f_series, f_direct)
    return _ensure_scalar_return(f, x)


def pressure_chandrasekhar(rho_gcm3, ye):
    """Pressione esatta di Chandrasekhar (erg/cm^3): P(rho, ye) = _P0_CGS*f(x(rho,ye)).

    Formula: P(x) = _P0_CGS*f(x), dove x = fermi_x(rho_gcm3, ye) e f e'
    la formula esatta di Chandrasekhar (con branching per evitare
    cancellazione catastrofica a x piccolo).

    Questa e' la EOS ESATTA, da usare come riferimento in tutti i
    confronti (mai `pressure_non_relativistic`/`pressure_ultrarelativistic`
    come "verita'" oltre i rispettivi limiti asintotici).

    Vettorizzata: accetta sia scalari float sia np.ndarray per rho_gcm3.
    """
    x = fermi_x(rho_gcm3, ye)
    return _P0_CGS * chandrasekhar_f(x)


def pressure_non_relativistic(rho_gcm3, ye):
    """Limite non-relativistico (x->0) della pressione di degenerazione.

    Formula diretta in n_e (NON passa da fermi_x/chandrasekhar_f):
        P_NR = ((3*pi^2)^(2/3)/5) * (hbar^2/m_e) * n_e^(5/3)
    con n_e = ye*rho_gcm3/M_U_G. Corrisponde a P ∝ rho^(5/3) (n=3/2).
    """
    _validate_rho(rho_gcm3)
    _validate_ye(ye)

    n_e = ye * np.asarray(rho_gcm3, dtype=float) / M_U_G
    return (_3_PI2 ** (2.0 / 3.0) / 5.0) * (HBAR_CGS**2 / M_ELECTRON_G) * n_e ** (5.0 / 3.0)


def k_ultrarelativistic(ye):
    """Costante K_UR(ye) tale che P_UR = K_UR(ye)*rho^(4/3).

    K_UR(ye) = ((3*pi^2)^(1/3)/4) * hbar*c * (ye/M_U_G)^(4/3)
    """
    _validate_ye(ye)
    return (_3_PI2 ** (1.0 / 3.0) / 4.0) * HBAR_CGS * C_LIGHT_CGS * (ye / M_U_G) ** (4.0 / 3.0)


def pressure_ultrarelativistic(rho_gcm3, ye):
    """Limite ultra-relativistico (x->inf) della pressione di degenerazione.

    P_UR = K_UR(ye) * rho_gcm3^(4/3). Corrisponde a P ∝ rho^(4/3) (n=3).
    Riusa `k_ultrarelativistic`, non duplica la formula del prefattore.
    """
    _validate_rho(rho_gcm3)
    k_ur = k_ultrarelativistic(ye)
    return k_ur * np.asarray(rho_gcm3, dtype=float) ** (4.0 / 3.0)


def _gamma1_of_x(x):
    """Indice adiabatico Gamma1, funzione privata pure-x (nome fissato).

    Gamma1(x) = (8*x^5)/(3*chandrasekhar_f(x)*sqrt(1+x^2))

    Chiama `chandrasekhar_f(x)` cosi' com'e', branching incluso: la
    stabilita' numerica del ramo a serie per x piccolo si propaga
    correttamente al rapporto (numeratore 8*x^5 e denominatore
    3*f(x)*sqrt(1+x^2) entrambi ben condizionati quando f(x) e' calcolata
    con la serie); nessuna nuova cancellazione viene introdotta nel
    quoziente stesso.

    Necessaria (anche se non nella lista pubblica concordata col coder)
    per poter testare gli asintoti di Gamma1 "in forma pura su x", come
    richiesto dal piano — convenzione di naming coerente con i privati gia'
    presenti in lane_emden.py (_lane_emden_rhs, _theta_zero_event).
    """
    x_arr = np.asarray(x, dtype=float)
    f_x = chandrasekhar_f(x_arr)
    gamma1 = (8.0 * x_arr**5) / (3.0 * f_x * np.sqrt(1.0 + x_arr**2))
    return _ensure_scalar_return(gamma1, x)


def gamma1_chandrasekhar(rho_gcm3, ye):
    """Indice adiabatico esatto Gamma1(rho, ye) per EOS di Chandrasekhar.

    Gamma1 = dlog(P) / dlog(rho) |_s = (8*x^5) / (3*f(x)*sqrt(1+x^2)),
    dove x = fermi_x(rho_gcm3, ye).

    Limiti asintotici: Gamma1 -> 5/3 per x->0 (non-relativistico),
    Gamma1 -> 4/3 per x->inf (ultra-relativistico).

    Vettorizzata: accetta sia scalari float sia np.ndarray per rho_gcm3.
    """
    x = fermi_x(rho_gcm3, ye)
    return _gamma1_of_x(x)


def n_eff_chandrasekhar(rho_gcm3, ye):
    """Indice politropico effettivo locale: n_eff = 1/(Gamma1 - 1).

    Limiti attesi: n_eff -> 3/2 per x->0 (Gamma1->5/3), n_eff -> 3 per
    x->inf (Gamma1->4/3) — transizione EOS non-relativistica ->
    ultra-relativistica.
    """
    gamma1 = gamma1_chandrasekhar(rho_gcm3, ye)
    return 1.0 / (gamma1 - 1.0)


def chandrasekhar_mass_msun(ye):
    """Massa di Chandrasekhar NEWTONIANA CLASSICA (politropo n=3), in Msun.

    ATTENZIONE (vincolo di modello, CLAUDE.md/piano Step 3): questo e' il
    valore Newtoniano classico per un politropo n=3 con EOS completamente
    degenere a T=0, indipendente da rho_c — NON e' "la" soglia definitiva
    di instabilita' GR-corretta. Quella resta esclusivamente il limite di
    Oppenheimer-Volkoff (TOV), responsabilita' dello Step 6.

    Riusa direttamente `collasso.lane_emden.solve_lane_emden(3.0)` (Step 2)
    per (xi1, theta'(xi1)) del politropo n=3 — mai una costante tabulata a
    mano. Il valore atteso xi1≈6.89685 e' lo stesso gia' validato contro
    letteratura in Step 2 (test_xi1_literature_values).

    M_Ch_g = 4*pi*(K_UR(ye)/(pi*G_CGS))^1.5 * (-xi1^2*theta'(xi1))
    """
    _validate_ye(ye)

    solution = solve_lane_emden(3.0)
    # Guardia difensiva: per n=3 con xi_max di default (50.0) l'evento
    # theta==0 scatta sempre entro il dominio (xi1 finito atteso, ~6.9),
    # quindi questo ramo non dovrebbe mai attivarsi in pratica — ma il
    # controllo resta obbligatorio, stesso principio del check sol.success
    # di Step 2 (nessun risultato costruito su un presupposto non
    # verificato).
    if solution.xi1 is None:
        raise RuntimeError(
            "chandrasekhar_mass_msun: nessuna superficie finita trovata per "
            "il politropo n=3 (xi1 is None) — atteso non accadere mai con "
            "xi_max di default, controllo comunque obbligatorio."
        )

    k_ur = k_ultrarelativistic(ye)
    geom_term = -(solution.xi1**2) * solution.dtheta_dxi_at_xi1

    m_ch_g = 4.0 * math.pi * (k_ur / (math.pi * G_CGS)) ** 1.5 * geom_term
    return m_ch_g / M_SUN_G


def chandrasekhar_k_deviation_pct(rho_c_gcm3, ye, n, alpha_cm):
    """Scarto percentuale fra la costante politropica K IMPLICITA in un
    profilo di Lane-Emden scalato per massa (`collasso.lane_emden.
    physical_profile`, che fissa `alpha_cm` per far tornare la massa
    target, non la costante K) e la costante K della EOS ESATTA di
    Chandrasekhar alla stessa densita' centrale/n:

        K_implicito = alpha_cm^2 * 4*pi*G_CGS / ((n+1) * rho_c_gcm3^(1/n - 1))
        K_eos = pressure_chandrasekhar(rho_c_gcm3, ye) / rho_c_gcm3^(1 + 1/n)
        ritorna (K_implicito/K_eos - 1) * 100

    Perche' esiste questa funzione (ciclo "Integrazione catalogo reale",
    successivo a Step 8): per una stella con massa_nucleo_msun SOPRA la
    massa di Chandrasekhar (vedi `chandrasekhar_mass_msun`), NESSUN vero
    equilibrio idrostatico esiste con questa EOS a nessuna densita'
    centrale (la massa di un tale equilibrio e' sempre < M_Ch, per
    qualunque rho_c, anche arbitrariamente grande) — quindi il profilo
    costruito da `physical_profile` per raggiungere quella massa non puo'
    MAI essere un vero equilibrio: implica una costante K leggermente
    "piu' rigida" (K_implicito > K_eos) di quella reale della EOS. Questo
    scarto NON e' un errore di modellazione: e' la conseguenza fisica
    diretta e attesa del fatto che il nucleo e' supra-Chandrasekhar — una
    misura quantitativa di QUANTO lo e' (lo scarto cresce con la frazione
    massa_nucleo/M_Ch). Va sempre riportato esplicitamente nell'output per
    stella, non solo internamente (vedi `collasso.pipeline.SimulationResult`
    e `scripts/run_simulation.py`).

    Solleva:
        ValueError: propagato da `_validate_rho`/`_validate_ye` se
            rho_c_gcm3/ye non validi; se n <= 0 (indice politropico non
            fisico per questa formula, che divide per n).
    """
    _validate_rho(rho_c_gcm3)
    _validate_ye(ye)
    if n <= 0:
        raise ValueError(f"n={n} deve essere positivo per chandrasekhar_k_deviation_pct.")

    k_implicito = (alpha_cm**2) * 4.0 * math.pi * G_CGS / ((n + 1.0) * rho_c_gcm3 ** (1.0 / n - 1.0))
    p_c = pressure_chandrasekhar(rho_c_gcm3, ye)
    k_eos = p_c / rho_c_gcm3 ** (1.0 + 1.0 / n)

    return (k_implicito / k_eos - 1.0) * 100.0


def _rho_from_x(x, ye):
    """Inversione in forma chiusa x -> rho (formula fissata dal planner,
    NON root-finding): rho(x, ye) = M_U_G/(ye*3*pi^2) *
    (x*M_ELECTRON_G*C_LIGHT_CGS/HBAR_CGS)**3.

    Utile per costruire griglie a x fissato nello script demo e nei test
    (es. valutare l'EOS esattamente a x=1e-4 o x=1e4).
    """
    _validate_ye(ye)
    x_arr = np.asarray(x, dtype=float)
    if np.any(x_arr < 0):
        raise ValueError(f"x deve essere >= 0; trovato valore minimo={np.min(x_arr)}")

    rho = (M_U_G / (ye * _3_PI2)) * (x_arr * M_ELECTRON_G * C_LIGHT_CGS / HBAR_CGS) ** 3
    return _ensure_scalar_return(rho, x)
