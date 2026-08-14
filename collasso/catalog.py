"""
collasso.catalog — catalogo dei progenitori stellari per la simulazione del
collasso gravitazionale (Step 1).

Contiene:
- la dataclass `Progenitor`, con validazione hard dei parametri fisici
  contro i range vincolanti fissati nel piano dello Step 1 (STATUS.md,
  "Piano ciclo corrente (Step 1) — REV. 2");
- funzioni di caricamento del catalogo da CSV (`load_catalog_from_csv`,
  `load_reference_catalog`) e di selezione di una stella (`get_progenitor_by_id`).

Limiti di modello dichiarati (vedi anche CLAUDE.md):
- `n_politropico` è un singolo valore fisso per stella: è un'approssimazione
  dell'ultimo istante prima del collasso, NON una vera transizione EOS
  non-relativistica -> ultra-relativistica a indice variabile lungo il
  raggio (quella resta un limite di modello dichiarato anche dopo lo
  Step 3, che fornisce solo l'indice locale n_eff(rho, ye)).
- `raggio_iniziale_km` deve sempre provenire da letteratura O essere
  dichiarato esplicitamente "derivato internamente" in `fonte`, MAI un
  numero arbitrario non giustificato.

Nota sul catalogo di riferimento attuale (ciclo "Integrazione catalogo
reale", successivo a Step 8 — vedi STATUS.md e
`data/progenitors_reference.csv` per il dettaglio completo per campo):
`massa_nucleo_msun` e `ye` sono dati di letteratura (O'Connor & Ott 2011,
ApJ 730:70, Tabella 1, dati Woosley/Heger/Weaver 2002); `n_politropico` e
`raggio_iniziale_km` sono DERIVATI INTERNAMENTE dal progetto stesso
(`collasso.eos.n_eff_chandrasekhar` + `collasso.lane_emden`) a partire
dalla massa tabulata e da una densità centrale scelta come valore fisico
plausibile (non da letteratura) — quindi i 4 parametri SONO ora mutuamente
consistenti come singolo politropo di Lane-Emden by construction (a
differenza del vecchio catalogo placeholder, dove non lo erano). Limite
fisico verificato esplicitamente: le tre masse reali del nucleo sono tutte
sopra la massa di Chandrasekhar per ye=0.495 (M_Ch≈1.427 Msun) — non esiste
quindi, per nessuna delle tre stelle, un vero equilibrio idrostatico a EOS
di Chandrasekhar esatta; il profilo derivato va inteso come condizione
iniziale della dinamica (Step 4), non come soluzione di equilibrio.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

# Directory dati di default (root_progetto/data)
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_REFERENCE_CATALOG_PATH = _DATA_DIR / "progenitors_reference.csv"

# --- Range fisici vincolanti fissati dal planner per lo Step 1 -------------
YE_MIN, YE_MAX = 0.42, 0.50
N_POLITROPICO_MIN, N_POLITROPICO_MAX = 1.5, 3.0
MASSA_NUCLEO_MIN_MSUN, MASSA_NUCLEO_MAX_MSUN = 1.3, 2.0


@dataclass
class Progenitor:
    """Parametri di una stella progenitrice pre-collasso.

    Campi:
        id: identificativo della stella (es. "s15", "s20", "s25").
        massa_zams_msun: massa iniziale ZAMS del progenitore (massa "di
            targa" della stella, es. 15/20/25 Msun). NON coincide con la
            massa del nucleo pre-collasso.
        massa_nucleo_msun: massa del nucleo pre-collasso (parametro chiave
            per Step 2/3); deve rispettare il criterio di plausibilità
            Chandrasekhar, 1.3-2.0 Msun.
        densita_centrale_gcm3: densita centrale del nucleo (g/cm^3),
            parametro chiave per il solver di Lane-Emden (Step 2). Nel
            catalogo di riferimento attuale non e' un valore di
            letteratura (vedi nota di modulo e `fonte` per il dettaglio).
        raggio_iniziale_km: raggio iniziale del nucleo (km). Nel catalogo
            di riferimento attuale e' derivato internamente da Lane-Emden
            (Step 2) + EOS di Chandrasekhar (Step 3), non preso da
            letteratura — vedi `fonte` per il dettaglio per stella.
        ye: frazione di elettroni (adimensionale), range vincolante
            0.42-0.50 per nuclei di ferro/silicio pre-collasso.
        n_politropico: indice politropico (adimensionale), range
            vincolante 1.5-3.0; valore fisso per stella (limite di modello
            dichiarato, vedi docstring del modulo). Nel catalogo di
            riferimento attuale e' derivato da
            `collasso.eos.n_eff_chandrasekhar` alla densita' centrale
            scelta, non un valore arbitrario.
        fonte: riferimento bibliografico esplicito, con provenienza
            distinta per campo (letteratura vs. derivato internamente vs.
            valore fisico plausibile) — vedi
            `data/progenitors_reference.csv` per la convenzione completa.
        nota_proxy: stringa vuota (default) per un vero progenitore del
            catalogo; NON vuota SOLO quando questa voce non rappresenta
            un modello calcolato specificamente per l'oggetto nominato,
            ma un modello generico riusato come proxy dichiarato
            esplicitamente (es. "betelgeuse": riusa s20WHW02 come modello
            di massa iniziale piu' vicina alla stima reale, non una
            ricostruzione della struttura interna specifica della
            stella). Quando non vuota, `collasso.pipeline.
            run_full_simulation` aggiunge un disclaimer aggiuntivo (il
            nono) con questo testo — l'etichettatura "proxy" compare
            quindi sia nel catalogo sia nell'output finale, mai solo in
            un posto solo.
    """

    id: str
    massa_zams_msun: float
    massa_nucleo_msun: float
    densita_centrale_gcm3: float
    raggio_iniziale_km: float
    ye: float
    n_politropico: float
    fonte: str
    nota_proxy: str = ""

    def __post_init__(self) -> None:
        if not (YE_MIN <= self.ye <= YE_MAX):
            raise ValueError(
                f"Progenitor '{self.id}': ye={self.ye} fuori dal range "
                f"fisico plausibile [{YE_MIN}, {YE_MAX}] per nuclei di "
                f"ferro/silicio pre-collasso."
            )

        if not (N_POLITROPICO_MIN <= self.n_politropico <= N_POLITROPICO_MAX):
            raise ValueError(
                f"Progenitor '{self.id}': n_politropico={self.n_politropico} "
                f"fuori dal range fisico plausibile "
                f"[{N_POLITROPICO_MIN}, {N_POLITROPICO_MAX}] per EOS degenere."
            )

        if not (MASSA_NUCLEO_MIN_MSUN <= self.massa_nucleo_msun <= MASSA_NUCLEO_MAX_MSUN):
            raise ValueError(
                f"Progenitor '{self.id}': massa_nucleo_msun="
                f"{self.massa_nucleo_msun} fuori dal range di plausibilita "
                f"Chandrasekhar [{MASSA_NUCLEO_MIN_MSUN}, "
                f"{MASSA_NUCLEO_MAX_MSUN}] Msun."
            )

        if self.massa_zams_msun <= 0:
            raise ValueError(
                f"Progenitor '{self.id}': massa_zams_msun="
                f"{self.massa_zams_msun} deve essere positiva."
            )

        if self.massa_nucleo_msun <= 0:
            raise ValueError(
                f"Progenitor '{self.id}': massa_nucleo_msun="
                f"{self.massa_nucleo_msun} deve essere positiva."
            )

        if self.densita_centrale_gcm3 <= 0:
            raise ValueError(
                f"Progenitor '{self.id}': densita_centrale_gcm3="
                f"{self.densita_centrale_gcm3} deve essere positiva."
            )

        if self.raggio_iniziale_km <= 0:
            raise ValueError(
                f"Progenitor '{self.id}': raggio_iniziale_km="
                f"{self.raggio_iniziale_km} deve essere positivo."
            )

        if self.massa_nucleo_msun > self.massa_zams_msun:
            raise ValueError(
                f"Progenitor '{self.id}': massa_nucleo_msun="
                f"{self.massa_nucleo_msun} non puo superare "
                f"massa_zams_msun={self.massa_zams_msun}."
            )


def load_catalog_from_csv(path: str | Path) -> list[Progenitor]:
    """Carica un catalogo di progenitori da un file CSV.

    Parsing generico ed estensibile, pensato per poter leggere in futuro
    il catalogo completo Sukhbold/Woosley & Heger senza riscrivere la
    funzione. Le righe che iniziano con '#' (commento bibliografico in
    testa al file) vengono saltate esplicitamente.

    Le colonne attese nel CSV corrispondono ai campi della dataclass
    `Progenitor`: id, massa_zams_msun, massa_nucleo_msun,
    densita_centrale_gcm3, raggio_iniziale_km, ye, n_politropico, fonte.
    `nota_proxy` e' opzionale (colonna assente o vuota -> ""), per
    retrocompatibilita' con CSV che non la includono.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Catalogo non trovato: {path}")

    progenitors: list[Progenitor] = []
    with path.open(newline="", encoding="utf-8") as f:
        # Filtra manualmente le righe di commento in testa/sparse nel file,
        # cosi' csv.DictReader vede solo intestazione + righe dati.
        righe_utili = (riga for riga in f if not riga.lstrip().startswith("#"))
        reader = csv.DictReader(righe_utili)
        for row in reader:
            # Salta eventuali righe vuote residue.
            if all((v is None or str(v).strip() == "") for v in row.values()):
                continue
            progenitors.append(
                Progenitor(
                    id=row["id"].strip(),
                    massa_zams_msun=float(row["massa_zams_msun"]),
                    massa_nucleo_msun=float(row["massa_nucleo_msun"]),
                    densita_centrale_gcm3=float(row["densita_centrale_gcm3"]),
                    raggio_iniziale_km=float(row["raggio_iniziale_km"]),
                    ye=float(row["ye"]),
                    n_politropico=float(row["n_politropico"]),
                    fonte=row["fonte"].strip(),
                    nota_proxy=(row.get("nota_proxy") or "").strip(),
                )
            )

    return progenitors


def load_reference_catalog() -> list[Progenitor]:
    """Carica il catalogo di riferimento (7 voci: s15/s20/s25/s30/s35/s40,
    tutte dalla stessa Tabella 1 di O'Connor & Ott 2011, piu' "betelgeuse",
    un PROXY dichiarato che riusa s20WHW02 — vedi `Progenitor.nota_proxy`)
    da `data/progenitors_reference.csv`. Provenienza mista per campo (dati
    di letteratura + valori derivati internamente) — vedi nota di modulo e
    il file CSV per il dettaglio completo, non piu' interamente
    placeholder.
    """
    return load_catalog_from_csv(_REFERENCE_CATALOG_PATH)


def get_progenitor_by_id(catalog: list[Progenitor], id: str) -> Progenitor:
    """Restituisce il progenitore con l'id indicato dal catalogo fornito.

    Solleva `ValueError` con messaggio chiaro (inclusi gli id disponibili)
    se l'id non e' presente nel catalogo.
    """
    for progenitor in catalog:
        if progenitor.id == id:
            return progenitor

    id_disponibili = [p.id for p in catalog]
    raise ValueError(
        f"Nessun progenitore con id '{id}' nel catalogo. "
        f"Id disponibili: {id_disponibili}"
    )
