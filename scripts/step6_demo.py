"""
scripts/step6_demo.py — demo Step 6 + Retrofit Step 6: limite di massa di
Oppenheimer-Volkoff (TOV), correzione del bug fisico (rho -> epsilon/c^2),
EOS realistica a politropi a tratti (SLy/APR4), classificazione del remnant
per la stella scelta s20.

Contenuti (vedi STATUS.md, piano Step 6 e "Retrofit Step 6 — EOS
realistica"): TRE confronti, per disaggregare onestamente il fix del bug
fisico dalla fisica nucleare realistica:
    1. Step 6 originale/bug (gas di neutroni liberi, rho come sorgente):
       riferimento storico, NON fisicamente corretto.
    2. Gas di neutroni liberi CORRETTO (epsilon/c^2 come sorgente): valida
       l'integratore TOV contro il valore storico di Oppenheimer & Volkoff
       1939 (~0.7 Msun).
    3. SLy/APR4 (EOS realistica, Read et al. 2009): confrontate con
       letteratura (2.05/2.20 Msun) e con l'osservazione PSR J0740+6620
       (Fonseca et al. 2021).
Riclassificazione di s20 con le tre soglie di massa (gas libero corretto,
SLy, APR4). Disclaimer aggiornato.

Nessun plot Matplotlib (riservato a Step 7). Output filtrato (regola
CLAUDE.md): solo le righe di riepilogo elencate sopra, nessun log grezzo.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collasso.catalog import get_progenitor_by_id, load_reference_catalog
from collasso.eos import chandrasekhar_mass_msun
from collasso.eos_piecewise_polytrope import build_apr4, build_sly
from collasso.remnant import classify_remnant
from collasso.tov import (
    RHO_C_GRID_REALISTIC_CGS,
    find_ov_mass_limit,
    find_ov_mass_limit_legacy_bug_rho_source,
)

# Valore storico di riferimento (Oppenheimer & Volkoff, Phys. Rev. 55, 374,
# 1939), per il confronto di scarto percentuale stampato sotto.
M_OV_STORICO_MSUN = 0.7

# Valori di letteratura per SLy/APR4 (Read et al. 2009 e successivi).
M_MAX_LITERATURE_MSUN = {"SLy": 2.05, "APR4": 2.20}

# Osservazione PSR J0740+6620 (Fonseca et al. 2021), massa piu' pesante
# misurata con precisione di una stella di neutroni, usata come confronto
# osservativo (non un vincolo di modello del progetto).
M_PSR_J0740_MSUN = 2.08
M_PSR_J0740_SIGMA_MSUN = 0.07


def main() -> None:
    print("=== Step 6 + Retrofit demo: limite TOV, EOS realistica, classificazione remnant ===")

    # --- Confronto 1: path storico/bacato di Step 6 (rho come sorgente) ----
    rho_c_legacy, m_max_legacy_msun, seq_legacy = find_ov_mass_limit_legacy_bug_rho_source()
    print(
        f"[1/3] Step 6 ORIGINALE (bug fisico noto, rho come sorgente in dP/dr e dm/dr, "
        f"NON fisicamente corretto): M_max={m_max_legacy_msun:.4f} Msun a "
        f"rho_c={rho_c_legacy:.3e} g/cm^3 - preservato SOLO per riferimento storico."
    )

    # --- Confronto 2: gas di neutroni liberi CORRETTO (epsilon/c^2) --------
    rho_c_corrected, m_max_corrected_msun, seq_corrected = find_ov_mass_limit()
    scarto_pct_corrected = 100.0 * abs(m_max_corrected_msun - M_OV_STORICO_MSUN) / M_OV_STORICO_MSUN
    print(
        f"[2/3] Gas di neutroni liberi CORRETTO (epsilon/c^2 come sorgente, fix del bug "
        f"fisico): M_max={m_max_corrected_msun:.4f} Msun a rho_c={rho_c_corrected:.3e} g/cm^3 "
        f"(griglia: {len(seq_corrected)} punti, rho_c in [1e14, 1e17] g/cm^3); "
        f"scarto dal valore storico O&V 1939 (~{M_OV_STORICO_MSUN} Msun) = {scarto_pct_corrected:.1f}% "
        f"(era {100.0 * abs(m_max_legacy_msun - M_OV_STORICO_MSUN) / M_OV_STORICO_MSUN:.1f}% col bug) "
        "-> valida l'integratore TOV stesso."
    )

    # --- Confronto 3: EOS realistica a politropi a tratti (SLy/APR4) -------
    risultati_realistici = {}
    for nome, build_fn in [("SLy", build_sly), ("APR4", build_apr4)]:
        eos = build_fn()
        rho_c_max, m_max_msun, seq = find_ov_mass_limit(RHO_C_GRID_REALISTIC_CGS, eos=eos)
        risultati_realistici[nome] = m_max_msun
        scarto_pct_lit = 100.0 * abs(m_max_msun - M_MAX_LITERATURE_MSUN[nome]) / M_MAX_LITERATURE_MSUN[nome]
        print(
            f"[3/3] EOS {nome} (Read et al. 2009): M_max={m_max_msun:.4f} Msun a "
            f"rho_c={rho_c_max:.3e} g/cm^3 (griglia: {len(seq)} punti, rho_c in "
            f"[3.16e14, 1e16] g/cm^3); scarto da letteratura (~{M_MAX_LITERATURE_MSUN[nome]} "
            f"Msun) = {scarto_pct_lit:.1f}%"
        )

    scarto_psr_sly_sigma = abs(risultati_realistici["SLy"] - M_PSR_J0740_MSUN) / M_PSR_J0740_SIGMA_MSUN
    scarto_psr_apr4_sigma = abs(risultati_realistici["APR4"] - M_PSR_J0740_MSUN) / M_PSR_J0740_SIGMA_MSUN
    print(
        f"Confronto osservativo: PSR J0740+6620 = {M_PSR_J0740_MSUN}+-{M_PSR_J0740_SIGMA_MSUN} Msun "
        f"(Fonseca et al. 2021) -> SLy scarto={scarto_psr_sly_sigma:.1f} sigma, "
        f"APR4 scarto={scarto_psr_apr4_sigma:.1f} sigma."
    )

    # --- Classificazione del remnant, stella scelta s20 ---------------------
    catalogo = load_reference_catalog()
    stella = get_progenitor_by_id(catalogo, "s20")

    m_ch_msun = chandrasekhar_mass_msun(stella.ye)

    classificazione_gas_corretto = classify_remnant(stella.massa_nucleo_msun, m_ch_msun, m_max_corrected_msun)
    classificazione_sly = classify_remnant(stella.massa_nucleo_msun, m_ch_msun, risultati_realistici["SLy"])
    classificazione_apr4 = classify_remnant(stella.massa_nucleo_msun, m_ch_msun, risultati_realistici["APR4"])

    print(
        f"Stella scelta s20: massa_nucleo_msun={stella.massa_nucleo_msun:.4f} Msun, "
        f"ye={stella.ye:.2f} -> M_Chandrasekhar(ye)={m_ch_msun:.4f} Msun (Step 3, Newtoniana classica)."
    )
    print(
        f"Riclassificazione remnant s20 con le tre soglie TOV:\n"
        f"  - gas neutroni liberi CORRETTO (M_TOV={m_max_corrected_msun:.4f} Msun) "
        f"-> '{classificazione_gas_corretto}'\n"
        f"  - SLy (M_TOV={risultati_realistici['SLy']:.4f} Msun) -> '{classificazione_sly}'\n"
        f"  - APR4 (M_TOV={risultati_realistici['APR4']:.4f} Msun) -> '{classificazione_apr4}'"
    )

    # --- Disclaimer obbligatorio ----------------------------------------------
    print(
        "DISCLAIMER (limiti di modello dichiarati): il gas di neutroni liberi "
        "NON interagenti (formalismo originale di Oppenheimer & Volkoff 1939) "
        "trascura le interazioni nucleari forti e sottostima sistematicamente "
        "il vero limite di massa osservativo delle stelle di neutroni "
        "(~2-2.2 Msun) - resta utile SOLO come validazione dell'integratore "
        "TOV contro il valore storico O&V 1939. SLy e APR4 sono fit "
        "fenomenologici a politropi a tratti (Read et al. 2009), NON EOS "
        "nucleari a molti corpi complete: accurati entro ~1% sulle relazioni "
        "massa-raggio delle rispettive EOS nucleari originali, non su ogni "
        "osservabile microscopica. Raffreddamento neutrini semplificato, "
        "rotazione e campo magnetico trascurati (limiti gia' esclusi a monte, "
        "invariati dagli step precedenti)."
    )
    print(
        f"NOTA sul caso specifico s20 (massa_nucleo_msun={stella.massa_nucleo_msun:.2f}): con "
        "l'EOS realistica SLy/APR4 la soglia TOV sale da ~0.71 a ~2.0-2.2 Msun, "
        "quindi la classificazione cambia rispetto al gas di neutroni liberi "
        "semplificato - il valore riportato sopra con SLy/APR4 e' la stima "
        "PIU' affidabile prodotta finora da questo progetto per questo "
        "progenitore, pur restando un fit fenomenologico e non una EOS "
        "nucleare completa."
    )


if __name__ == "__main__":
    main()
