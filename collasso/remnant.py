"""
collasso.remnant — classificazione del tipo di remnant compatto (Step 6).

Utility pura, nessuna dipendenza da `collasso.tov` (logica testabile in
isolamento, come richiesto dal piano Step 6): confronta la massa del nucleo
(o del residuo, a seconda del chiamante) contro le due soglie di
instabilita' gia' calcolate altrove nel progetto:
    - massa di Chandrasekhar (`collasso.eos.chandrasekhar_mass_msun`,
      Newtoniana classica, Step 3);
    - limite di Oppenheimer-Volkoff (`collasso.tov.find_ov_mass_limit`,
      Step 6).

Vincolo di modello (CLAUDE.md, tabella vincoli, riga "Classificazione
remnant"): la soglia definitiva GR-corretta e' SEMPRE il limite di
Oppenheimer-Volkoff (TOV), non la massa di Chandrasekhar Newtoniana (usata
qui solo come prima soglia, sotto la quale il collasso non porta a una
stella di neutroni ma si arresta a una nana bianca).
"""

from __future__ import annotations


def classify_remnant(massa_msun: float, m_chandrasekhar_msun: float, m_tov_msun: float) -> str:
    """Classifica il tipo di remnant compatto in base a tre soglie di massa.

    - `"white_dwarf"` se `massa_msun < m_chandrasekhar_msun`;
    - `"neutron_star"` se `m_chandrasekhar_msun <= massa_msun < m_tov_msun`;
    - `"black_hole"` se `massa_msun >= m_tov_msun`.

    Nessuna validazione fisica aggiuntiva sui parametri (funzione pura,
    utility di confronto): il chiamante e' responsabile della coerenza
    fisica dei tre valori passati (es. m_chandrasekhar_msun < m_tov_msun in
    condizioni fisicamente sensate).
    """
    if massa_msun < m_chandrasekhar_msun:
        return "white_dwarf"
    if massa_msun < m_tov_msun:
        return "neutron_star"
    return "black_hole"
