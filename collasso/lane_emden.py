"""
collasso.lane_emden — solver dell'equazione di Lane-Emden per il profilo di
equilibrio politropico iniziale del nucleo stellare (Step 2).

Equazione risolta (forma standard, decisione vincolante concordata con
l'utente):
    (1/xi^2) d/dxi(xi^2 dtheta/dxi) = -theta^n
con condizioni al contorno theta(0)=1, theta'(0)=0.

Vincoli di modello per questo step (da CLAUDE.md e piano Step 2, non
negoziabili):
- Gravita' SOLO newtoniana: nessuna correzione relativistica introdotta qui
  (responsabilita' esclusiva dello Step 5).
- `n` (indice politropico) e' un valore FISSO per ogni singola soluzione:
  nessuna transizione a indice variabile non-relativistico ->
  ultra-relativistico (responsabilita' esclusiva dello Step 3, EOS di
  Chandrasekhar).
- Nessun termine di raffreddamento neutrini, nessuna rotazione/campo
  magnetico (limiti gia' dichiarati, invariati da Step 1).

Metodo numerico:
- la singolarita' in xi=0 (per il termine 2/xi * dtheta/dxi) viene aggirata
  innescando l'integrazione a un piccolo xi0, con condizioni iniziali date
  dallo sviluppo in serie troncato di theta e dtheta/dxi attorno a xi=0;
- da xi0 in poi si integra con `scipy.integrate.solve_ivp` (RK45, tolleranze
  strette) il sistema del primo ordine in (theta, phi=dtheta/dxi);
- la superficie xi1 (primo zero di theta, cioe' il bordo della stella nelle
  unita' adimensionali di Lane-Emden) viene individuata tramite un evento
  terminale di `solve_ivp` (theta==0, direction=-1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from collasso.constants import KM_CM, M_SUN_G

# Innesco della serie a piccolo xi0 (valore fissato dal planner, non a
# discrezione del coder).
XI0_DEFAULT = 1e-4


@dataclass
class LaneEmdenSolution:
    """Soluzione numerica dell'equazione di Lane-Emden per un dato `n`.

    Campi:
        n: indice politropico usato per questa soluzione (fisso, vedi
            docstring del modulo).
        xi: griglia di ascisse adimensionali su cui e' campionata la
            soluzione (da xi0 a xi1, o a xi_max se xi1 non esiste entro
            xi_max).
        theta: valori di theta sulla griglia `xi`. IMPORTANTE: questo array
            NON ha il clipping a >=0 applicato (il clip `max(theta, 0.0)' e'
            usato SOLO internamente, dentro il RHS dell'ODE al punto 1.3 del
            piano e nel calcolo di `rho_gcm3` in `physical_profile`, MAI su
            questo array). Di conseguenza l'ultimo punto della griglia
            (xi=xi1, se esiste) puo' contenere un residuo numerico
            leggermente negativo (tipicamente ~1e-10) dovuto all'evento che
            interrompe l'integrazione appena dopo l'attraversamento dello
            zero. Codice a valle NON deve assumere theta>=0 ovunque su
            questo campo.
        dtheta_dxi: valori di dtheta/dxi sulla griglia `xi` (stesso dense
            output della soluzione, nessuna clippatura).
        xi1: ascissa del primo zero di theta (superficie della stella in
            unita' di Lane-Emden), oppure `None` se non esiste una
            superficie finita entro `xi_max` (atteso per n>=5, dove la
            soluzione analitica ha xi1 infinito). Nessuna eccezione viene
            sollevata in questo caso: e' un esito atteso e documentato.
        dtheta_dxi_at_xi1: dtheta/dxi valutata a xi1 (pendenza della
            soluzione alla superficie, ingrediente della massa totale del
            politropo), oppure `None` se `xi1` e' `None`.
    """

    n: float
    xi: np.ndarray
    theta: np.ndarray
    dtheta_dxi: np.ndarray
    xi1: float | None
    dtheta_dxi_at_xi1: float | None


@dataclass
class PhysicalProfile:
    """Profilo fisico (dimensionale) derivato da una `LaneEmdenSolution`.

    Campi:
        n: indice politropico della soluzione di Lane-Emden sottostante.
        rho_c_gcm3: densita' centrale usata per la conversione (g/cm^3).
        massa_msun: massa totale usata per fissare la scala (Msun) — deve
            essere sempre `massa_nucleo_msun` del catalogo quando questa
            funzione e' chiamata dallo script demo, mai `massa_zams_msun`.
        alpha_cm: fattore di scala radiale di Lane-Emden (cm), tale che
            r_cm = alpha_cm * xi.
        R_cm: raggio della superficie (xi1) in cm.
        R_km: raggio della superficie in km.
        r_cm: griglia radiale dimensionale (cm), r_cm = alpha_cm * xi.
        rho_gcm3: profilo di densita' dimensionale (g/cm^3),
            rho_c_gcm3 * max(theta, 0)^n (stesso clipping usato nel RHS
            dell'ODE, per coerenza).
    """

    n: float
    rho_c_gcm3: float
    massa_msun: float
    alpha_cm: float
    R_cm: float
    R_km: float
    r_cm: np.ndarray
    rho_gcm3: np.ndarray


def _lane_emden_rhs(xi: float, y: np.ndarray, n: float) -> list[float]:
    """RHS del sistema del primo ordine (theta, phi=dtheta/dxi).

    Il clipping `max(theta, 0.0)` e' obbligatorio e uniforme per ogni n
    (anche n=0,1, per coerenza di implementazione): evita basi negative
    elevate a esponente non intero (NaN/complesso) quando theta scende
    leggermente sotto zero per errore numerico vicino a xi1.
    """
    theta, phi = y
    dtheta_dxi = phi
    dphi_dxi = -(2.0 / xi) * phi - max(theta, 0.0) ** n
    return [dtheta_dxi, dphi_dxi]


def _theta_zero_event(xi: float, y: np.ndarray, n: float) -> float:
    """Evento terminale: theta == 0 (attraversamento decrescente).

    Utilizzato da `solve_ivp` per individuare la superficie della stella
    (primo zero di theta). La direzione di attraversamento è fissata a -1
    (decrescente) tramite l'attributo `direction` per garantire che il primo
    zero sia quello fisicamente rilevante.
    """
    return y[0]


_theta_zero_event.direction = -1.0
_theta_zero_event.terminal = True


def solve_lane_emden(
    n: float,
    xi0: float = XI0_DEFAULT,
    xi_max: float = 50.0,
    n_points: int = 2000,
) -> LaneEmdenSolution:
    """Risolve l'equazione di Lane-Emden per l'indice politropico `n`.

    Parametri:
        n: indice politropico (deve essere >= 0, fisso per questa
            soluzione — vedi vincoli di modello nel docstring del modulo).
        xi0: ascissa di innesco della serie (default 1e-4, valore fissato
            dal planner).
        xi_max: limite superiore di integrazione se non viene trovata una
            superficie finita (default 50.0; per n=5 va usato esplicitamente
            questo valore, coerente con le soluzioni analitiche concordate).
        n_points: numero di punti della griglia di output (default 2000).

    Ritorna:
        `LaneEmdenSolution` con la soluzione campionata su `n_points` punti
        tra xi0 e xi1 (o xi_max se xi1 non esiste entro xi_max).

    Solleva:
        ValueError: se n < 0 (indice politropico non fisico).
        RuntimeError: se l'integrazione di `solve_ivp` fallisce
            (`sol.success is False`), riportando `sol.message`.
    """
    if n < 0:
        raise ValueError(f"n={n} non fisico: l'indice politropico deve essere >= 0.")

    # Innesco in serie a xi0 (formule fissate dal planner, da usare esattamente
    # come scritte, non da ri-derivare):
    theta_xi0 = 1.0 - xi0**2 / 6.0 + n * xi0**4 / 120.0
    dtheta_dxi_xi0 = -xi0 / 3.0 + n * xi0**3 / 30.0

    sol = solve_ivp(
        fun=_lane_emden_rhs,
        t_span=(xi0, xi_max),
        y0=[theta_xi0, dtheta_dxi_xi0],
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
        dense_output=True,
        events=_theta_zero_event,
        args=(n,),
    )

    # Check esplicito ed incondizionato su sol.success, PRIMA di processare
    # l'evento o costruire LaneEmdenSolution: senza questo controllo, un
    # fallimento silenzioso di RK45 vicino alla rigidita' numerica a xi~xi1
    # produrrebbe una soluzione apparentemente valida ma costruita su
    # un'integrazione parziale/incompleta.
    if not sol.success:
        raise RuntimeError(f"integrazione Lane-Emden fallita per n={n}: {sol.message}")

    # Determinazione di xi1 e della pendenza alla superficie a partire
    # dall'evento terminale (se scattato entro xi_max).
    if len(sol.t_events[0]) > 0:
        xi1 = float(sol.t_events[0][0])
        dtheta_dxi_at_xi1 = float(sol.y_events[0][0][1])
    else:
        xi1 = None
        dtheta_dxi_at_xi1 = None

    xi_end = xi1 if xi1 is not None else xi_max
    xi = np.linspace(xi0, xi_end, n_points)
    theta, dtheta_dxi = sol.sol(xi)

    return LaneEmdenSolution(
        n=n,
        xi=xi,
        theta=theta,
        dtheta_dxi=dtheta_dxi,
        xi1=xi1,
        dtheta_dxi_at_xi1=dtheta_dxi_at_xi1,
    )


def theta_analytic(xi: np.ndarray, n: float) -> np.ndarray:
    """Soluzione analitica di Lane-Emden, solo per n=0 e n=1.

    n=0: theta = 1 - xi^2/6
    n=1: theta = sin(xi)/xi

    Per qualunque altro n solleva `NotImplementedError`: il confronto per
    n=5 (theta finito ma xi1 infinito) e' gestito separatamente e SOLO
    inline in demo/test, non qui, per non creare l'illusione di un xi1
    finito che non esiste per n=5.
    """
    xi = np.asarray(xi)
    if n == 0:
        return 1.0 - xi**2 / 6.0
    if n == 1:
        return np.sin(xi) / xi
    raise NotImplementedError(
        f"theta_analytic non implementata per n={n}: soluzione analitica "
        "chiusa disponibile solo per n=0 e n=1 in questo modulo. Il caso "
        "n=5 (theta finito, xi1 infinito) va confrontato inline in "
        "demo/test, non tramite questa funzione."
    )


def xi1_analytic(n: float) -> float:
    """xi1 analitico (primo zero di theta), solo per n=0 e n=1.

    n=0: xi1 = sqrt(6)
    n=1: xi1 = pi

    Per qualunque altro n solleva `NotImplementedError` (vedi
    `theta_analytic`; per n=5 xi1 e' infinito e va trattato separatamente).
    """
    if n == 0:
        return math.sqrt(6.0)
    if n == 1:
        return math.pi
    raise NotImplementedError(
        f"xi1_analytic non implementata per n={n}: valore chiuso disponibile "
        "solo per n=0 e n=1. Per n=5 xi1 e' infinito (nessuna superficie "
        "finita), gestito separatamente in demo/test."
    )


def physical_profile(
    solution: LaneEmdenSolution,
    rho_c_gcm3: float,
    massa_msun: float,
) -> PhysicalProfile:
    """Converte una `LaneEmdenSolution` adimensionale in un profilo fisico.

    Usa la coppia primaria (rho_c_gcm3, massa_msun) per fissare la scala
    radiale `alpha_cm`, invertendo la relazione massa-densita'-geometria di
    Lane-Emden:
        massa_msun*M_SUN_G = 4*pi*alpha_cm^3*rho_c_gcm3*(-xi1^2*dtheta_dxi_at_xi1)
    Nessuna correzione relativistica, nessuna dipendenza da G_CGS (non
    richiesta da questa relazione in questo step — vedi collasso.constants).

    `massa_msun` deve sempre essere `massa_nucleo_msun` del catalogo quando
    questa funzione e' chiamata dallo script demo (mai `massa_zams_msun`).

    Solleva:
        ValueError: se `solution.xi1` e' `None` (nessuna superficie finita
            entro xi_max, atteso per n>=5) o se il termine geometrico
            `(-xi1^2*dtheta_dxi_at_xi1)` non e' positivo (condizione non
            fisica, controllo difensivo).
    """
    if solution.xi1 is None:
        raise ValueError(
            "profilo fisico non definibile: nessuna superficie finita per "
            f"n={solution.n} entro xi_max, atteso per n>=5"
        )

    xi1 = solution.xi1
    dtheta_dxi_at_xi1 = solution.dtheta_dxi_at_xi1
    geom_term = -(xi1**2) * dtheta_dxi_at_xi1

    if geom_term <= 0:
        raise ValueError(
            "profilo fisico non definibile: termine geometrico "
            f"(-xi1^2*dtheta_dxi_at_xi1)={geom_term} non positivo "
            "(condizione non fisica)."
        )

    massa_g = massa_msun * M_SUN_G
    alpha_cm = (massa_g / (4.0 * math.pi * rho_c_gcm3 * geom_term)) ** (1.0 / 3.0)

    R_cm = alpha_cm * xi1
    R_km = R_cm / KM_CM

    r_cm = alpha_cm * solution.xi
    # Stesso clipping usato nel RHS dell'ODE (punto 1.3), per coerenza:
    # evita densita' negative dovute al residuo numerico di theta vicino a xi1.
    rho_gcm3 = rho_c_gcm3 * np.maximum(solution.theta, 0.0) ** solution.n

    return PhysicalProfile(
        n=solution.n,
        rho_c_gcm3=rho_c_gcm3,
        massa_msun=massa_msun,
        alpha_cm=alpha_cm,
        R_cm=R_cm,
        R_km=R_km,
        r_cm=r_cm,
        rho_gcm3=rho_gcm3,
    )
