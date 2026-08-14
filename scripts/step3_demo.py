"""
scripts/step3_demo.py — demo Step 3: EOS di Chandrasekhar del gas di
elettroni degenere (non-relativistico -> ultra-relativistico).

Contenuti (vedi piano Step 3, STATUS.md, sezione "2. Script demo"):
- tabella P_esatta / P_NR / P_UR sulla griglia x=1e-4 -> 1e4 (17 punti,
  un punto a decade), con gli scarti relativi rispetto a P_esatta e la
  transizione dell'indice politropico effettivo n_eff;
- massa di Chandrasekhar (Newtoniana classica, politropo n=3) per ye=0.5
  e per i tre ye del catalogo di riferimento (s15/s20/s25), con confronto
  esplicito al valore "~1.4 Msun" citato in CLAUDE.md;
- disclaimer obbligatorio sul significato (Newtoniano classico, non
  GR-corretto) di `chandrasekhar_mass_msun`.

Output filtrato (regola CLAUDE.md): solo tabelle/righe di riepilogo,
nessun log grezzo di calcolo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Permette di eseguire lo script direttamente (python scripts/step3_demo.py),
# stesso pattern usato in scripts/step1_demo.py e scripts/step2_demo.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collasso.catalog import load_reference_catalog
from collasso.eos import (
    _rho_from_x,
    chandrasekhar_mass_msun,
    n_eff_chandrasekhar,
    pressure_chandrasekhar,
    pressure_non_relativistic,
    pressure_ultrarelativistic,
)

# Valore di riferimento citato in CLAUDE.md (tabella dei vincoli fisici).
MASSA_CHANDRASEKHAR_RIFERIMENTO_MSUN = 1.4


def _scarto_percentuale(valore: float, riferimento: float) -> float:
    """Scarto percentuale: 100 * |valore - riferimento| / riferimento."""
    return 100.0 * abs(valore - riferimento) / riferimento


def _tabella_pressioni(ye: float = 0.5) -> None:
    """Tabella P_esatta/P_NR/P_UR e transizione di n_eff sulla griglia x."""
    x_grid = np.logspace(-4, 4, 17)
    rho_grid = _rho_from_x(x_grid, ye)

    p_esatta = pressure_chandrasekhar(rho_grid, ye)
    p_nr = pressure_non_relativistic(rho_grid, ye)
    p_ur = pressure_ultrarelativistic(rho_grid, ye)
    n_eff = n_eff_chandrasekhar(rho_grid, ye)

    scarto_nr_pct = 100.0 * np.abs(p_nr - p_esatta) / p_esatta
    scarto_ur_pct = 100.0 * np.abs(p_ur - p_esatta) / p_esatta

    print(f"=== Tabella P(rho): esatta vs limiti NR/UR (ye={ye}) ===")
    print(
        f"{'rho[g/cm3]':>12} {'x':>10} {'P_esatta':>12} {'P_NR':>12} "
        f"{'scarto_NR%':>11} {'P_UR':>12} {'scarto_UR%':>11} {'n_eff':>7}"
    )
    for i in range(len(x_grid)):
        marcatore = " <-- transizione (x~1)" if 0.3 <= x_grid[i] <= 3.0 else ""
        print(
            f"{rho_grid[i]:12.3e} {x_grid[i]:10.1e} {p_esatta[i]:12.4e} "
            f"{p_nr[i]:12.4e} {scarto_nr_pct[i]:11.2f} {p_ur[i]:12.4e} "
            f"{scarto_ur_pct[i]:11.2f} {n_eff[i]:7.3f}{marcatore}"
        )

    print(
        "NOTA: in prossimita' di x~1 (righe marcate sopra) P_esatta si "
        "discosta significativamente sia da P_NR sia da P_UR: e' il "
        "comportamento atteso nella regione di transizione, non un errore "
        "di calcolo."
    )
    print(
        f"Transizione n_eff lungo la griglia: da {n_eff[0]:.3f} (x piccolo, "
        f"atteso ~1.5) a {n_eff[-1]:.3f} (x grande, atteso ~3.0)."
    )


def _massa_chandrasekhar() -> None:
    """M_Ch(ye) per ye=0.5 e per i ye del catalogo di riferimento."""
    print()
    print("=== Massa di Chandrasekhar M_Ch(ye) -- politropo n=3 Newtoniano ===")

    m_ch_05 = chandrasekhar_mass_msun(0.5)
    scarto_05_pct = _scarto_percentuale(m_ch_05, MASSA_CHANDRASEKHAR_RIFERIMENTO_MSUN)
    print(
        f"ye=0.50 (riferimento): M_Ch={m_ch_05:.4f} Msun, scarto da "
        f"~1.4 Msun (CLAUDE.md) = {scarto_05_pct:.2f}%"
    )

    catalogo = load_reference_catalog()
    for stella in catalogo:
        m_ch = chandrasekhar_mass_msun(stella.ye)
        scarto_pct = _scarto_percentuale(m_ch, MASSA_CHANDRASEKHAR_RIFERIMENTO_MSUN)
        print(
            f"id={stella.id} ye={stella.ye:.2f}: M_Ch={m_ch:.4f} Msun, "
            f"scarto da ~1.4 Msun (CLAUDE.md) = {scarto_pct:.2f}%"
        )

    print(
        "DISCLAIMER: chandrasekhar_mass_msun e' il valore NEWTONIANO "
        "CLASSICO per un politropo n=3 (EOS completamente degenere, T=0), "
        "indipendente da rho_c. NON e' la soglia GR-corretta di "
        "instabilita': quella resta esclusivamente il limite di "
        "Oppenheimer-Volkoff (TOV), responsabilita' dello Step 6."
    )


def main() -> None:
    _tabella_pressioni(ye=0.5)
    _massa_chandrasekhar()


if __name__ == "__main__":
    main()
