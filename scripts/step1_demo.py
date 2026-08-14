"""
scripts/step1_demo.py — demo dello Step 1: carica il catalogo di riferimento
dei progenitori stellari, stampa una tabella compatta e i parametri della
stella scelta (default: 20 Msun, caso intermedio).

Output volutamente compatto (regola CLAUDE.md sul filtraggio dell'output
numerico verboso): nessun log grezzo, solo tabella riassuntiva e riepilogo
finale.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permette di eseguire lo script direttamente (python scripts/step1_demo.py)
# senza dover installare il pacchetto o impostare PYTHONPATH a mano.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collasso.catalog import Progenitor, get_progenitor_by_id, load_reference_catalog

STELLA_SCELTA_ID = "s20"  # caso intermedio (20 Msun), default per gli step successivi


def stampa_tabella(catalog: list[Progenitor]) -> None:
    intestazione = (
        f"{'id':<5}{'M_ZAMS[Msun]':>14}{'M_nucleo[Msun]':>16}"
        f"{'rho_c[g/cm3]':>15}{'R_i[km]':>10}{'Ye':>7}{'n':>6}"
    )
    print(intestazione)
    print("-" * len(intestazione))
    for p in catalog:
        print(
            f"{p.id:<5}{p.massa_zams_msun:>14.2f}{p.massa_nucleo_msun:>16.2f}"
            f"{p.densita_centrale_gcm3:>15.2e}{p.raggio_iniziale_km:>10.1f}"
            f"{p.ye:>7.3f}{p.n_politropico:>6.2f}"
        )


def stampa_stella_scelta(p: Progenitor) -> None:
    print()
    print(f"Stella scelta per gli step successivi: id='{p.id}'")
    print(f"  massa ZAMS            : {p.massa_zams_msun:.2f} Msun")
    print(f"  massa nucleo pre-coll. : {p.massa_nucleo_msun:.2f} Msun")
    print(f"  densita centrale      : {p.densita_centrale_gcm3:.3e} g/cm^3")
    print(f"  raggio iniziale       : {p.raggio_iniziale_km:.1f} km")
    print(f"  Ye                    : {p.ye:.3f}")
    print(f"  n politropico         : {p.n_politropico:.2f}")
    print(f"  fonte                 : {p.fonte}")


def main() -> None:
    catalog = load_reference_catalog()

    print(f"Catalogo di riferimento caricato: {len(catalog)} progenitori (placeholder).")
    print()
    stampa_tabella(catalog)

    stella_scelta = get_progenitor_by_id(catalog, STELLA_SCELTA_ID)
    stampa_stella_scelta(stella_scelta)


if __name__ == "__main__":
    main()
